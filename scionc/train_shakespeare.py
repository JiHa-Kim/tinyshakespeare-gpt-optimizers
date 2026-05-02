import argparse
import math
import time
from contextlib import nullcontext
from dataclasses import asdict, fields
from pathlib import Path

import torch

from scionc.ulmos import (
    ColNormULMO,
    GramNewtonSchulzULMO,
    HiddenSVDFilterULMO,
    RowNormULMO,
    SignULMO,
    StreamingSVDULMO,
    init_colnorm_,
    init_rownorm_,
    init_sign_,
    init_spectral_,
)
from scionc.optim import ScionC, lionk_S
from scionc.models import (
    GPT,
    MLP,
    BatchSource,
    CausalSelfAttention,
    CharDataset,
    GPTConfig,
    maybe_download_tiny_shakespeare,
)
from scionc.probes.convergence import ConvergenceProbe
from scionc.probes.line import (
    apply_line_scale,
    capture_params,
    capture_rng,
    finish_line_snapshot,
    line_curve_text,
    line_probe_text,
    parse_line_scales,
    restore_rng,
)
from scionc.probes.optimizer_stats import (
    accumulate_step_stats,
    capture_step_stats,
    consume_step_stats,
)

DEFAULT_BETA_HALF_LIFE = 156_489.1137547854
DEFAULT_SHRINK_HALF_LIVES = {
    "embed": 318_760.11959306244,
    "hidden": 967_726.9243144159,
    "out": 3_239_039.393395033,
}
DEFAULT_STEADY_RADII = {
    "embed": 1.0,
    "hidden": 3.0,
    "out": 10.0,
}


def sync_now(device: torch.device) -> float:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return time.perf_counter()


def cuda_memory_stats(device: torch.device) -> dict[str, float]:
    if device.type != "cuda":
        return {}
    return {
        "alloc_gb": torch.cuda.memory_allocated(device) / 1e9,
        "reserved_gb": torch.cuda.memory_reserved(device) / 1e9,
        "max_reserved_gb": torch.cuda.max_memory_reserved(device) / 1e9,
        "total_gb": torch.cuda.get_device_properties(device).total_memory / 1e9,
    }


def cuda_memory_text(device: torch.device) -> str:
    stats = cuda_memory_stats(device)
    if not stats:
        return ""
    return (
        f" | cuda_alloc {stats['alloc_gb']:.2f}G"
        f" | cuda_reserved {stats['reserved_gb']:.2f}G"
        f" | cuda_max_reserved {stats['max_reserved_gb']:.2f}G"
    )


def step_stats_text(stats: dict[str, dict]) -> str:
    if not stats:
        return ""
    parts = []
    for name, values in stats.items():
        parts.append(
            f"{name}:cos={values['cos']:.3f},"
            f"u/p={values['update_param_rms']:.2e},"
            f"u/g={values['update_grad_rms']:.2e},"
            f"g/p={values['grad_param_rms']:.2e},"
            f"xg={values['param_grad_cos']:.3f},"
            f"xu={values['param_update_cos']:.3f},"
            f"ga/r={values['grad_abs_rms']:.3f},"
            f"ua/r={values['update_abs_rms']:.3f},"
            f"gk={values['grad_kurtosis']:.2e},"
            f"uk={values['update_kurtosis']:.2e}"
        )
    return " | step_stats " + "; ".join(parts)


def amp_ctx(amp_dtype: torch.dtype | None):
    return (
        torch.autocast(device_type="cuda", dtype=amp_dtype)
        if amp_dtype is not None
        else nullcontext()
    )


def resolve_schedule(
    max_steps: int, warmup_steps: int, decay_steps: int
) -> tuple[int, int, int]:
    if max_steps <= 0:
        raise ValueError(f"invalid max_steps: {max_steps}")
    warmup_steps = max(0, min(warmup_steps, max_steps))
    decay_steps = max(0, min(decay_steps, max_steps - warmup_steps))
    stable_steps = max_steps - warmup_steps - decay_steps
    return warmup_steps, stable_steps, decay_steps


def schedule_at_step(
    step: int,
    max_steps: int,
    peak: float,
    floor: float,
    warmup_steps: int,
    decay_steps: int,
) -> float:
    warmup_steps, stable_steps, decay_steps = resolve_schedule(
        max_steps, warmup_steps, decay_steps
    )

    if warmup_steps > 0 and step < warmup_steps:
        return peak * (step + 1) / warmup_steps

    decay_start = warmup_steps + stable_steps
    if decay_steps == 0 or step < decay_start:
        return peak
    if decay_steps == 1:
        return floor

    progress = (step - decay_start) / (decay_steps - 1)
    progress = min(max(progress, 0.0), 1.0)
    if peak > 0.0 and floor > 0.0:
        return 1.0 / ((1.0 - progress) / peak + progress / floor)
    return peak + (floor - peak) * progress


@torch.inference_mode()
def estimate_loss(
    model: GPT,
    source: BatchSource,
    eval_iters: int,
    amp_dtype: torch.dtype | None,
    splits=("train", "val"),
):
    was_training = model.training
    model.eval()
    out = {}
    with amp_ctx(amp_dtype):
        for split in splits:
            total = 0.0
            for _ in range(eval_iters):
                _, loss = model(*source.get(split))
                total += float(loss)
            out[split] = total / eval_iters
    model.train(was_training)
    return out


@torch.inference_mode()
def estimate_val_loss(
    model: GPT, source: BatchSource, eval_iters: int, amp_dtype: torch.dtype | None
) -> float:
    return estimate_loss(model, source, eval_iters, amp_dtype, splits=("val",))["val"]


def resolve_group_step_scale(args, group: str) -> tuple[float, float]:
    peak = getattr(args, f"step_scale_{group}", None)
    if peak is None:
        peak = args.step_scale
    floor = getattr(args, f"min_step_scale_{group}", None)
    if floor is None:
        floor = args.min_step_scale
    if floor is None:
        floor = 0.1 * peak
    if peak < 0.0:
        raise ValueError(f"invalid {group} step scale: {peak}")
    if floor < 0.0:
        raise ValueError(f"invalid {group} minimum step scale: {floor}")
    return peak, floor


def resolve_group_rho(args, group: str) -> float:
    rho = getattr(args, f"rho_{group}", None)
    if rho is None:
        rho = args.rho
    if rho is None:
        rho = DEFAULT_STEADY_RADII[group]
    if rho <= 0.0:
        raise ValueError(f"invalid {group} steady radius: {rho}")
    return rho


def count_increment(args) -> int:
    return args.batch_size * args.block_size * args.grad_accum


def halving_factor(delta_tau: float, half_life: float, name: str) -> float:
    if delta_tau <= 0.0:
        raise ValueError(f"invalid count increment: {delta_tau}")
    if half_life <= 0.0:
        raise ValueError(f"invalid {name}: {half_life}")
    if math.isinf(half_life):
        return 1.0
    return 2.0 ** (-delta_tau / half_life)


def resolve_group_shrink_half_life(
    args, group: str
) -> float:
    half_life = getattr(args, f"shrink_half_life_{group}", None)
    if half_life is None:
        half_life = args.shrink_half_life
    if half_life is None:
        half_life = DEFAULT_SHRINK_HALF_LIVES[group]
    if half_life <= 0.0:
        raise ValueError(f"invalid {group} shrink half-life: {half_life}")
    return half_life


def apply_auto_group_step_scales(
    opt, summary: dict[str, dict[str, float]], args
) -> str:
    parts = []
    for group in opt.param_groups:
        name = group.get("name", "group")
        l1 = summary.get(name, {}).get("l1")
        if l1 is None or not math.isfinite(l1) or l1 <= 0.0:
            continue
        old = group.get("auto_l1", l1)
        l1 = args.auto_l1_beta * old + (1.0 - args.auto_l1_beta) * l1
        group["auto_l1"] = l1

        eta = args.auto_action_scale / l1
        unit = eta_unit(group)
        step_scale = eta / unit
        step_scale = min(
            max(step_scale, group.get("min_step_scale", 0.0)),
            group.get("max_step_scale", step_scale),
        )
        group["peak_step_scale"] = step_scale
        parts.append(f"{name}={step_scale:.3e}")
    return "auto_step_scale " + ", ".join(parts) if parts else ""


def make_hidden_ulmo(args, work_dtype: torch.dtype):
    if args.hidden_ulmo == "gram-ns":
        return GramNewtonSchulzULMO(steps=args.pe_steps, work_dtype=work_dtype)
    if args.hidden_ulmo == "svd-filter":
        return HiddenSVDFilterULMO(
            steps=args.spi_steps,
            ridge=args.spi_ridge,
            refresh_interval=args.spi_refresh_interval,
            refresh_threshold=args.spi_refresh_threshold,
            iteration=args.spi_iteration,
            filter_ridge=args.filter_ridge,
        )
    return StreamingSVDULMO(
        steps=args.spi_steps,
        ridge=args.spi_ridge,
        refresh_interval=args.spi_refresh_interval,
        refresh_threshold=args.spi_refresh_threshold,
        iteration=args.spi_iteration,
    )


def make_embed_ulmo(args):
    if args.embed_ulmo == "sign":
        return SignULMO()
    if args.embed_ulmo == "rownorm":
        return RowNormULMO()
    return ColNormULMO(transpose=True)


def make_out_ulmo(args):
    if args.out_ulmo == "colnorm":
        return ColNormULMO(transpose=True)
    if args.out_ulmo == "rownorm":
        return RowNormULMO()
    return SignULMO()


def hidden_params(model: GPT) -> list[torch.Tensor]:
    skip = {id(model.tok_emb.weight), id(model.lm_head.weight)}
    return [p for p in model.parameters() if p.requires_grad and id(p) not in skip]


def eta_unit(group: dict, eps: float = 1e-12) -> float:
    shrink = float(group["shrink"])
    if not (0.0 < shrink < 1.0):
        raise ValueError(
            f"cannot derive additive eta from shrink={shrink}; "
            "use a finite shrink half-life"
        )
    cu2 = float(group.get("rms_cu2", 1.0))
    q = float(group.get("rms_q", 1.0))
    s = float(group.get("rms_s", 1.0))
    if cu2 <= 0.0:
        raise ValueError(f"invalid RMS atom scale: {cu2}")
    if q <= 0.0:
        raise ValueError(f"invalid RMS keep fraction: {q}")
    if s <= 0.0:
        raise ValueError(f"invalid RMS momentum factor: {s}")
    variance = q * max(1.0 - shrink * shrink, eps) / (cu2 * s)
    return float(group["rho"]) * math.sqrt(variance)


def additive_eta(group: dict, step_scale: float) -> float:
    return step_scale * eta_unit(group)


@torch.no_grad()
def init_like_ulmo_(p: torch.Tensor, ulmo, radius: float) -> None:
    if isinstance(ulmo, ColNormULMO):
        init_colnorm_(p, radius=radius, transpose=ulmo.transpose)
    elif isinstance(ulmo, RowNormULMO):
        init_rownorm_(p, radius=radius, transpose=ulmo.transpose)
    elif isinstance(ulmo, SignULMO):
        init_sign_(p, radius=radius, transpose=ulmo.transpose)
    elif isinstance(ulmo, (GramNewtonSchulzULMO, StreamingSVDULMO)):
        init_spectral_(p, radius=radius, input_like=getattr(ulmo, "input_like", False))
    else:
        raise TypeError(f"unsupported ULMO init: {type(ulmo).__name__}")


@torch.no_grad()
def init_from_actions_(groups: list[dict]) -> None:
    for group in groups:
        radius = float(group["rho"])
        ulmo = group["ulmo"]
        for p in group["params"]:
            init_like_ulmo_(p, ulmo, radius)


def optimizer_group(
    name: str,
    params: list[torch.Tensor],
    ulmo,
    args,
    delta_tau: int,
    rms_s: float,
) -> dict | None:
    if not params:
        return None
    peak_step_scale, min_step_scale = resolve_group_step_scale(args, name)
    rho = resolve_group_rho(args, name)
    shrink_half_life = resolve_group_shrink_half_life(args, name)
    shrink = halving_factor(delta_tau, shrink_half_life, f"{name}_shrink_half_life")
    group = {
        "name": name,
        "params": params,
        "ulmo": ulmo,
        "rho": rho,
        "step_scale": peak_step_scale,
        "peak_step_scale": peak_step_scale,
        "max_step_scale": peak_step_scale,
        "min_step_scale": min_step_scale,
        "shrink": shrink,
        "shrink_half_life": shrink_half_life,
        "beta_half_life": args.beta_half_life,
        "count_increment": delta_tau,
        "rms_cu2": 1.0,
        "rms_q": 1.0,
        "rms_s": rms_s,
    }
    group["lr"] = additive_eta(group, peak_step_scale)
    return {
        **group,
    }


@torch.no_grad()
def build_optimizer(model: GPT, args, device: torch.device):
    delta_tau = count_increment(args)
    memory_beta = halving_factor(delta_tau, args.beta_half_life, "beta_half_life")
    rms_s = lionk_S(args.readout_mu, memory_beta, nesterov=True)
    work_dtype = torch.float16 if device.type == "cuda" else torch.float32
    groups = [
        optimizer_group(
            "embed",
            [model.tok_emb.weight],
            make_embed_ulmo(args),
            args,
            delta_tau,
            rms_s,
        ),
        optimizer_group(
            "hidden",
            hidden_params(model),
            make_hidden_ulmo(args, work_dtype),
            args,
            delta_tau,
            rms_s,
        ),
        optimizer_group(
            "out",
            [model.lm_head.weight],
            make_out_ulmo(args),
            args,
            delta_tau,
            rms_s,
        ),
    ]
    groups = [group for group in groups if group is not None]
    init_from_actions_(groups)

    hidden_group = next(group for group in groups if group["name"] == "hidden")
    return ScionC(
        groups,
        lr=hidden_group["lr"],
        readout_mu=args.readout_mu,
        memory_beta=memory_beta,
    )


def group_ulmo(opt, name: str, cls):
    for group in opt.param_groups:
        if group.get("name") == name and isinstance(group.get("ulmo"), cls):
            return group["ulmo"]
    return None


def hidden_cov_ulmo(opt):
    return group_ulmo(opt, "hidden", HiddenSVDFilterULMO)


def register_hidden_cov_hooks(model: GPT, ulmo):
    handles = []

    def covariance(x):
        flat = x.detach().reshape(-1, x.size(-1))
        return (flat.mT @ flat).float(), flat.size(0)

    def make_linear_hook(weight):
        def hook(_module, inputs):
            if not _module.training or not torch.is_grad_enabled():
                return
            cov, count = covariance(inputs[0])
            ulmo.add_covariance(weight, cov, count)

        return hook

    def make_mlp_hook(module: MLP):
        def hook(_module, inputs):
            if not _module.training or not torch.is_grad_enabled():
                return
            cov, count = covariance(inputs[0])
            ulmo.add_covariance(module.gate.weight, cov, count)
            ulmo.add_covariance(module.up.weight, cov, count)

        return hook

    def make_qkv_hook(module: CausalSelfAttention):
        def hook(_module, inputs):
            if not _module.training or not torch.is_grad_enabled():
                return
            cov, count = covariance(inputs[0])
            ulmo.add_covariance(module.q.weight, cov, count)
            ulmo.add_covariance(module.k.weight, cov, count)
            ulmo.add_covariance(module.v.weight, cov, count)

        return hook

    for module in model.modules():
        if isinstance(module, CausalSelfAttention):
            handles.append(module.register_forward_pre_hook(make_qkv_hook(module)))
            handles.append(
                module.proj.register_forward_pre_hook(
                    make_linear_hook(module.proj.weight)
                )
            )
        elif isinstance(module, MLP):
            handles.append(module.register_forward_pre_hook(make_mlp_hook(module)))
            handles.append(
                module.down.register_forward_pre_hook(
                    make_linear_hook(module.down.weight)
                )
            )
    return handles


def save_checkpoint(path: Path, model: GPT, dataset: CharDataset, args):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "model_cfg": asdict(model.cfg),
            "chars": dataset.chars,
            "args": vars(args),
        },
        path,
    )


def save_eval_checkpoint(
    path: Path,
    step: int,
    val_loss: float,
    model: GPT,
    dataset: CharDataset,
    args,
):
    if args.save_interval <= 0 or step % args.save_interval != 0:
        return
    eval_path = path.with_name(
        f"{path.stem}_step{step:05d}_val{val_loss:.4f}{path.suffix}"
    )
    save_checkpoint(eval_path, model, dataset, args)


def save_final_checkpoint(path: Path, model: GPT, dataset: CharDataset, args):
    final_path = path.with_name(f"{path.stem}_final{path.suffix}")
    save_checkpoint(final_path, model, dataset, args)


def load_checkpoint(path: Path, device: torch.device):
    ckpt = torch.load(path, map_location=device)
    chars = ckpt["chars"]
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}
    cfg_keys = {field.name for field in fields(GPTConfig)}
    cfg = {k: v for k, v in ckpt["model_cfg"].items() if k in cfg_keys}
    model = GPT(GPTConfig(**cfg)).to(device)
    model.load_state_dict(ckpt["model"])
    return model, stoi, itos


def maybe_compile(
    model: GPT,
    source: BatchSource,
    args,
    amp_dtype: torch.dtype | None,
    device: torch.device,
):
    if not (args.compile and hasattr(torch, "compile")):
        return model, 0.0
    model = torch.compile(model)
    xb, yb = source.get("train")
    t0 = sync_now(device)
    model.zero_grad(set_to_none=True)
    with amp_ctx(amp_dtype):
        _, loss = model(xb, yb)
    loss.backward()
    model.zero_grad(set_to_none=True)
    return model, sync_now(device) - t0


def train(args):
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.set_float32_matmul_precision("high")
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    amp_dtype = (
        torch.bfloat16
        if device.type == "cuda" and torch.cuda.is_bf16_supported()
        else None
    )
    line_curve_scales = parse_line_scales(args.line_curve_scales)
    if line_curve_scales:
        args.track_line_probe = True

    data_path = Path(args.data_path)
    maybe_download_tiny_shakespeare(data_path)
    dataset = CharDataset(data_path)

    raw_model = GPT(
        GPTConfig(
            vocab_size=len(dataset.chars),
            block_size=args.block_size,
            n_layer=args.n_layer,
            n_head=args.n_head,
            d_model=args.d_model,
            rope_base=args.rope_base,
            prenorm=args.prenorm,
            dropout=args.dropout,
        )
    ).to(device)
    source = BatchSource(
        dataset.train, dataset.val, args.block_size, args.batch_size, device
    )
    opt = build_optimizer(raw_model, args, device)
    cov_ulmo = hidden_cov_ulmo(opt)
    if cov_ulmo is not None:
        register_hidden_cov_hooks(raw_model, cov_ulmo)
    conv_probe = (
        ConvergenceProbe(raw_model, opt, args)
        if args.track_convergence_stats or args.auto_step_scale_from_stats
        else None
    )
    if conv_probe is not None:
        conv_probe.register_hooks(raw_model)
        if args.compile:
            print("compile_disabled_for_convergence_stats")
            args.compile = False
    model, compile_seconds = maybe_compile(raw_model, source, args, amp_dtype, device)
    if cov_ulmo is not None:
        cov_ulmo.cov_accums.clear()
    if compile_seconds:
        print(f"compile_seconds {compile_seconds:.3f}")

    warmup_steps = (
        args.warmup_iters
        if args.warmup_iters >= 0
        else round(args.warmup_frac * args.max_iters)
    )
    decay_steps = (
        args.decay_iters
        if args.decay_iters >= 0
        else round(args.decay_frac * args.max_iters)
    )
    warmup_steps, stable_steps, decay_steps = resolve_schedule(
        args.max_iters, warmup_steps, decay_steps
    )
    effective_tokens = count_increment(args)
    first_group = opt.param_groups[0]
    readout_mu = first_group.get("readout_mu", args.readout_mu)
    memory_beta = first_group.get("memory_beta", math.nan)
    beta_half_life = first_group.get("beta_half_life", math.nan)

    group_schedule = []
    for group in opt.param_groups:
        name = group.get("name", "group")
        peak_step_scale = group.get("peak_step_scale", group["step_scale"])
        min_step_scale = group.get("min_step_scale", 0.0)
        peak_eta = additive_eta(group, peak_step_scale)
        min_eta = additive_eta(group, min_step_scale)
        rho = group.get("rho", math.nan)
        shrink = group.get("shrink", 1.0)
        shrink_half_life = group.get("shrink_half_life", math.inf)
        rms_s = group.get("rms_s", math.nan)
        group_schedule.append(
            f"{name}=(rho={rho:g},s={peak_step_scale:.3g}->{min_step_scale:.3g},"
            f"eta={peak_eta:.3e}->{min_eta:.3e},h_shrink={shrink_half_life:.3g},"
            f"zeta={shrink:.6f},S={rms_s:.3g})"
        )
    print(
        "schedule "
        f"warmup_steps={warmup_steps} stable_steps={stable_steps} decay_steps={decay_steps} "
        f"count_increment={effective_tokens} "
        f"beta_half_life={beta_half_life:.3g} beta={memory_beta:.6f} "
        f"readout_mu={readout_mu:.3g} "
        f"optimizer=scionc prenorm={args.prenorm} dropout={args.dropout:.3f} "
        f"hidden_ulmo={args.hidden_ulmo} "
        f"embed_ulmo={args.embed_ulmo} out_ulmo={args.out_ulmo} "
        f"qkv=split spi_iteration={args.spi_iteration} "
        f"eta_groups=" + ", ".join(group_schedule)
    )
    if args.track_line_probe and args.grad_accum != 1:
        print("line_probe_disabled_requires_grad_accum_1")
    if line_curve_scales:
        print("line_curve_scales " + ",".join(f"{x:g}" for x in line_curve_scales))

    total_opt_steps = 0
    best_val = float("inf")
    max_val = float("-inf")
    last_losses = {"train": float("nan"), "val": float("nan")}
    initial_val = None
    diverged = False
    diverge_reason = ""
    train_start = sync_now(device)
    step_stat_accum = {}

    for step in range(args.max_iters):
        current_etas = {}
        for group in opt.param_groups:
            peak_step_scale = group.get("peak_step_scale", group["step_scale"])
            min_step_scale = group.get("min_step_scale", 0.0)
            step_scale = schedule_at_step(
                step,
                args.max_iters,
                peak_step_scale,
                min_step_scale,
                warmup_steps,
                decay_steps,
            )
            group["step_scale"] = step_scale
            eta = additive_eta(group, step_scale)
            group["lr"] = eta
            current_etas[group.get("name", f"group{len(current_etas)}")] = eta
        eta = current_etas.get("hidden", next(iter(current_etas.values())))

        if step % args.eval_interval == 0 or step == args.max_iters - 1:
            train_loss = last_losses["train"]
            val_loss = estimate_val_loss(model, source, args.eval_iters, amp_dtype)
            last_losses["val"] = val_loss
            opt_stats = (
                consume_step_stats(step_stat_accum) if args.track_step_stats else {}
            )

            if not math.isfinite(val_loss):
                diverged, diverge_reason = True, "nonfinite_eval_loss"
            else:
                prev_best = best_val
                if initial_val is None:
                    initial_val = val_loss
                best_val = min(best_val, val_loss)
                max_val = max(max_val, val_loss)
                if step > 0 and val_loss > initial_val * args.diverge_mult:
                    diverged = True
                    diverge_reason = (
                        f"val_loss_exceeded_{args.diverge_mult:.2f}x_initial"
                    )
                if not args.no_save and val_loss < prev_best:
                    save_checkpoint(Path(args.out_path), raw_model, dataset, args)
                if not args.no_save:
                    save_eval_checkpoint(
                        Path(args.out_path), step, val_loss, raw_model, dataset, args
                    )
                    if step == args.max_iters - 1:
                        save_final_checkpoint(
                            Path(args.out_path), raw_model, dataset, args
                        )

            elapsed = max(sync_now(device) - train_start, 1e-9)
            mem_text = cuda_memory_text(device)
            opt_text = step_stats_text(opt_stats)
            print(
                f"step {step:5d} | eta {eta:.3e} | train {train_loss:.4f} | val {val_loss:.4f} | "
                f"best_val {best_val:.4f} | train_seconds {elapsed:.3f} | "
                f"tok/s {(total_opt_steps * effective_tokens) / elapsed:.0f}"
                f"{mem_text}{opt_text}"
            )
            memory = cuda_memory_stats(device)
            if (
                args.max_cuda_reserved_gb > 0
                and memory
                and memory["reserved_gb"] > args.max_cuda_reserved_gb
            ):
                diverged = True
                diverge_reason = (
                    f"cuda_reserved_{memory['reserved_gb']:.2f}G_exceeded_"
                    f"{args.max_cuda_reserved_gb:.2f}G"
                )
        if diverged:
            print(f"diverged {diverge_reason}")
            break

        line_active = (
            args.track_line_probe
            and args.grad_accum == 1
            and args.line_probe_interval > 0
            and step % args.line_probe_interval == 0
        )
        line_batch = None
        line_rng_before = None
        line_loss_before = None
        opt.zero_grad(set_to_none=True)
        if conv_probe is not None:
            conv_probe.start_step(step)
        train_loss = 0.0
        for micro_step in range(args.grad_accum):
            batch = source.get("train")
            if line_active and micro_step == 0:
                line_batch = batch
                line_rng_before = capture_rng(device)
            with amp_ctx(amp_dtype):
                _, loss = model(*batch)
                loss = loss / args.grad_accum
            loss_value = float(loss.detach())
            if not math.isfinite(loss_value):
                diverged, diverge_reason = True, "nonfinite_train_loss"
                break
            if line_active and micro_step == 0:
                line_loss_before = loss_value
            train_loss += loss_value
            loss.backward()
        if diverged:
            print(f"diverged {diverge_reason}")
            break
        last_losses["train"] = train_loss

        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(raw_model.parameters(), args.grad_clip)
        if conv_probe is not None:
            conv_text = conv_probe.capture(step, current_etas)
            if conv_text:
                print(conv_text)
                if (
                    args.auto_step_scale_from_stats
                    and warmup_steps <= step < warmup_steps + stable_steps
                ):
                    auto_text = apply_auto_group_step_scales(
                        opt, conv_probe.summary, args
                    )
                    if auto_text:
                        print(auto_text)
        line_params_before = (
            capture_params(raw_model.parameters())
            if line_active and line_curve_scales
            else None
        )
        stat_snapshot = (
            capture_step_stats(opt)
            if args.track_step_stats or line_active
            else None
        )
        opt.step()
        line_stats = {}
        if stat_snapshot is not None:
            if args.track_step_stats:
                accumulate_step_stats(step_stat_accum, stat_snapshot)
            if line_active:
                line_stat_accum = {}
                accumulate_step_stats(line_stat_accum, stat_snapshot)
                line_stats = consume_step_stats(line_stat_accum)
        total_opt_steps = step + 1
        if (
            line_active
            and line_batch is not None
            and line_rng_before is not None
            and line_loss_before is not None
        ):
            rng_after = capture_rng(device)
            if line_params_before is None:
                restore_rng(line_rng_before, device)
                with torch.no_grad(), amp_ctx(amp_dtype):
                    _, loss_after = model(*line_batch)
                restore_rng(rng_after, device)
                loss_after_value = float(loss_after.detach())
            else:
                snapshot = finish_line_snapshot(line_params_before)
                curve_losses = []
                for scale in line_curve_scales:
                    apply_line_scale(snapshot, scale)
                    restore_rng(line_rng_before, device)
                    with torch.no_grad(), amp_ctx(amp_dtype):
                        _, curve_loss = model(*line_batch)
                    curve_losses.append((scale, float(curve_loss.detach())))
                apply_line_scale(snapshot, 1.0)
                restore_rng(rng_after, device)
                loss_after_value = min(
                    curve_losses, key=lambda item: abs(item[0] - 1.0)
                )[1]
                curve_text = line_curve_text(step, curve_losses)
                if curve_text:
                    print(curve_text)
            line_text = line_probe_text(
                step, line_loss_before, loss_after_value, line_stats
            )
            if line_text:
                print(line_text)

    if not (args.skip_sample or diverged):
        y = raw_model.generate(
            torch.tensor(
                [dataset.encode(args.prompt or "\n")], dtype=torch.long, device=device
            ),
            max_new_tokens=args.sample_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
        )
        print("\n--- sample ---\n")
        print(dataset.decode(y[0].tolist()))

    for group in opt.param_groups:
        stats = getattr(group.get("ulmo"), "stats", None)
        if stats:
            print(
                f"{group.get('name', 'group')}_ulmo_stats "
                + " ".join(f"{k}={v}" for k, v in stats.items())
            )

    return {
        "best_val": best_val,
        "final_train": last_losses["train"],
        "final_val": last_losses["val"],
        "compile_seconds": compile_seconds,
        "initial_val": float("nan") if initial_val is None else initial_val,
        "max_val": max_val,
        "diverged": diverged,
        "diverge_reason": diverge_reason,
        "warmup_steps": warmup_steps,
        "stable_steps": stable_steps,
        "decay_steps": decay_steps,
    }


@torch.inference_mode()
def sample(args):
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    model, stoi, itos = load_checkpoint(Path(args.out_path), device)
    prompt = args.prompt or "\n"
    bad = [c for c in prompt if c not in stoi]
    if bad:
        raise ValueError(f"prompt contains unseen chars: {bad}")
    x = torch.tensor([[stoi[c] for c in prompt]], dtype=torch.long, device=device)
    texts = []
    for _ in range(args.sample_count):
        y = model.generate(
            x,
            max_new_tokens=args.sample_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
        )
        texts.append("".join(itos[int(i)] for i in y[0].tolist()))

    if args.sample_out:
        path = Path(args.sample_out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(sample_report(args, texts), encoding="utf-8")
        print(f"wrote_samples {path}")
        return

    for i, text in enumerate(texts, start=1):
        if len(texts) > 1:
            print(f"\n--- sample {i} ---\n")
        print(text)


def sample_report(args, texts: list[str]) -> str:
    prompt = args.prompt or "\\n"
    lines = [
        "# Sample Report",
        "",
        f"- checkpoint: `{args.out_path}`",
        f"- seed: `{args.seed}`",
        f"- prompt: `{prompt}`",
        f"- sample_tokens: `{args.sample_tokens}`",
        f"- temperature: `{args.temperature}`",
        f"- top_k: `{args.top_k}`",
        f"- sample_count: `{len(texts)}`",
        "",
    ]
    for i, text in enumerate(texts, start=1):
        lines.extend([f"## Sample {i}", "", "```text", text, "```", ""])
    return "\n".join(lines)


@torch.inference_mode()
def evaluate(args):
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    amp_dtype = (
        torch.bfloat16
        if device.type == "cuda" and torch.cuda.is_bf16_supported()
        else None
    )

    data_path = Path(args.data_path)
    maybe_download_tiny_shakespeare(data_path)
    dataset = CharDataset(data_path)
    model, _, _ = load_checkpoint(Path(args.out_path), device)
    source = BatchSource(
        dataset.train, dataset.val, model.cfg.block_size, args.batch_size, device
    )
    losses = estimate_loss(model, source, args.eval_iters, amp_dtype)
    print(
        f"eval_iters {args.eval_iters} | batch_size {args.batch_size} | "
        f"train {losses['train']:.4f} | val {losses['val']:.4f}"
        f"{cuda_memory_text(device)}"
    )


def make_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["train", "sample", "eval"], default="train")
    p.add_argument("--data-path", default="data/tiny_shakespeare.txt")
    p.add_argument("--out-path", default="out/scion_shakespeare.pt")
    p.add_argument("--device", default="")
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--compile", action=argparse.BooleanOptionalAction, default=True)

    p.add_argument("--block-size", type=int, default=256)
    p.add_argument("--batch-size", type=int, default=64, help="microbatch size")
    p.add_argument("--grad-accum", type=int, default=1)
    p.add_argument("--n-layer", type=int, default=6)
    p.add_argument("--n-head", type=int, default=6)
    p.add_argument("--d-model", type=int, default=384)
    p.add_argument("--rope-base", type=float, default=10000.0)
    p.add_argument("--prenorm", choices=["rmsnorm", "rmsball"], default="rmsnorm")
    p.add_argument("--dropout", type=float, default=0.15)

    p.add_argument("--max-iters", type=int, default=2000)
    p.add_argument("--eval-interval", type=int, default=100)
    p.add_argument("--eval-iters", type=int, default=50)
    p.add_argument("--grad-clip", type=float, default=0.0)
    p.add_argument("--diverge-mult", type=float, default=2.0)

    p.add_argument(
        "--warmup-iters", type=int, default=100, help="if >=0, overrides warmup-frac"
    )
    p.add_argument("--warmup-frac", type=float, default=0.0)
    p.add_argument(
        "--decay-iters", type=int, default=-1, help="if >=0, overrides decay-frac"
    )
    p.add_argument("--decay-frac", type=float, default=0.15)

    p.add_argument(
        "--step-scale",
        type=float,
        default=1.0,
        help="dimensionless peak action scale; raw additive eta is derived",
    )
    p.add_argument(
        "--step-scale-embed", dest="step_scale_embed", type=float, default=None
    )
    p.add_argument(
        "--step-scale-hidden", dest="step_scale_hidden", type=float, default=None
    )
    p.add_argument(
        "--step-scale-out",
        "--step-scale-unembed",
        dest="step_scale_out",
        type=float,
        default=None,
    )
    p.add_argument(
        "--min-step-scale",
        type=float,
        default=0.0,
        help="dimensionless decay floor for action scale",
    )
    p.add_argument(
        "--min-step-scale-embed",
        dest="min_step_scale_embed",
        type=float,
        default=None,
    )
    p.add_argument(
        "--min-step-scale-hidden",
        dest="min_step_scale_hidden",
        type=float,
        default=None,
    )
    p.add_argument(
        "--min-step-scale-out",
        "--min-step-scale-unembed",
        dest="min_step_scale_out",
        type=float,
        default=None,
    )
    p.add_argument(
        "--rho",
        type=float,
        default=None,
        help="steady-state radius for all optimizer groups",
    )
    p.add_argument("--rho-embed", dest="rho_embed", type=float, default=None)
    p.add_argument("--rho-hidden", dest="rho_hidden", type=float, default=None)
    p.add_argument(
        "--rho-out", "--rho-unembed", dest="rho_out", type=float, default=None
    )
    p.add_argument(
        "--beta-half-life",
        type=float,
        default=DEFAULT_BETA_HALF_LIFE,
        help="EMA memory half-life in processed tokens",
    )
    p.add_argument(
        "--readout-mu",
        type=float,
        default=1.0,
        help="dimensionless Nesterov readout blend",
    )
    p.add_argument(
        "--hidden-ulmo",
        choices=["streaming-svd", "svd-filter", "gram-ns"],
        default="gram-ns",
        help="hidden-matrix ULMO",
    )
    p.add_argument(
        "--embed-ulmo",
        choices=["colnorm", "sign", "rownorm"],
        default="colnorm",
        help="embedding-table ULMO",
    )
    p.add_argument(
        "--out-ulmo",
        choices=["sign", "colnorm", "rownorm"],
        default="sign",
        help="output-head ULMO",
    )
    p.add_argument("--pe-steps", type=int, default=5, help="Gram-NS coefficient steps")
    p.add_argument("--spi-steps", type=int, default=1)
    p.add_argument("--spi-ridge", type=float, default=1e-3)
    p.add_argument(
        "--spi-iteration",
        choices=["scqr2", "norm-power"],
        default="norm-power",
        help="streaming-SVD subspace iteration path",
    )
    p.add_argument("--filter-ridge", type=float, default=1e-3)
    p.add_argument("--spi-refresh-interval", type=int, default=100)
    p.add_argument("--spi-refresh-threshold", type=float, default=0.10)
    p.add_argument(
        "--shrink-half-life",
        type=float,
        default=None,
        help="direct shrink half-life in processed tokens for all groups",
    )
    p.add_argument(
        "--shrink-half-life-embed",
        dest="shrink_half_life_embed",
        type=float,
        default=None,
    )
    p.add_argument(
        "--shrink-half-life-hidden",
        dest="shrink_half_life_hidden",
        type=float,
        default=None,
    )
    p.add_argument(
        "--shrink-half-life-out",
        "--shrink-half-life-unembed",
        dest="shrink_half_life_out",
        type=float,
        default=None,
    )

    p.add_argument("--prompt", default="To be, or not to be")
    p.add_argument("--sample-tokens", type=int, default=400)
    p.add_argument("--sample-count", type=int, default=1)
    p.add_argument("--sample-out", default="")
    p.add_argument("--temperature", type=float, default=0.9)
    p.add_argument("--top-k", type=int, default=40)
    p.add_argument("--skip-sample", action="store_true")
    p.add_argument("--no-save", action="store_true")
    p.add_argument(
        "--save-interval",
        type=int,
        default=400,
        help="save eval checkpoints every N steps in addition to best/final",
    )
    p.add_argument(
        "--max-cuda-reserved-gb",
        type=float,
        default=0.0,
        help="abort if process CUDA reserved memory exceeds this limit",
    )
    p.add_argument(
        "--track-step-stats",
        action="store_true",
        help="accumulate optimizer group stats and print them on eval lines",
    )
    p.add_argument(
        "--track-convergence-stats",
        action="store_true",
        help="probe Gluon smoothness and spectral-ratio stats during training",
    )
    p.add_argument(
        "--track-line-probe",
        action="store_true",
        help="estimate same-batch eta aggressiveness with one extra forward",
    )
    p.add_argument(
        "--line-probe-interval",
        type=int,
        default=100,
        help="optimizer-step interval for same-batch line probes",
    )
    p.add_argument(
        "--line-curve-scales",
        default="",
        help="comma-separated update multipliers for expensive same-batch line curves",
    )
    p.add_argument(
        "--convergence-interval",
        type=int,
        default=50,
        help="optimizer-step interval for convergence probes",
    )
    p.add_argument(
        "--convergence-action-scale",
        dest="convergence_action_scale",
        type=float,
        default=0.5,
        help="target normalized action scale for L1-derived eta reports",
    )
    p.add_argument(
        "--convergence-probe",
        choices=["representative", "all"],
        default="representative",
        help="which parameters to include in convergence probes",
    )
    p.add_argument(
        "--auto-step-scale-from-stats",
        dest="auto_step_scale_from_stats",
        action="store_true",
        help="set group peak step scales from convergence L1 estimates after warmup",
    )
    p.add_argument(
        "--auto-action-scale",
        dest="auto_action_scale",
        type=float,
        default=1.25,
        help="target normalized action scale eta * L1",
    )
    p.add_argument(
        "--auto-l1-beta",
        dest="auto_l1_beta",
        type=float,
        default=0.8,
        help="EMA beta for L1 estimates used by --auto-step-scale-from-stats",
    )
    return p


def main():
    args = make_parser().parse_args()
    if args.mode == "train":
        train(args)
    elif args.mode == "sample":
        sample(args)
    else:
        evaluate(args)


if __name__ == "__main__":
    main()
