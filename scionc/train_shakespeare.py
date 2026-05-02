import argparse
import math
import time
from contextlib import nullcontext
from dataclasses import asdict, fields
from pathlib import Path

import torch

from scionc.lmos import (
    ColNormLMO,
    GramNewtonSchulzLMO,
    HiddenSVDFilterLMO,
    RowNormLMO,
    ScionC,
    SignLMO,
    StreamingSVDSpectralLMO,
    init_colnorm_,
    init_sign_,
    init_spectral_,
)
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


def lr_at_step(
    step: int,
    max_steps: int,
    lr: float,
    min_lr: float,
    warmup_steps: int,
    decay_steps: int,
) -> float:
    warmup_steps, stable_steps, decay_steps = resolve_schedule(
        max_steps, warmup_steps, decay_steps
    )

    if warmup_steps > 0 and step < warmup_steps:
        return lr * (step + 1) / warmup_steps

    decay_start = warmup_steps + stable_steps
    if decay_steps == 0 or step < decay_start:
        return lr
    if decay_steps == 1:
        return min_lr

    progress = (step - decay_start) / (decay_steps - 1)
    progress = min(max(progress, 0.0), 1.0)
    if lr > 0.0 and min_lr > 0.0:
        return 1.0 / ((1.0 - progress) / lr + progress / min_lr)
    return lr + (min_lr - lr) * progress


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


@torch.no_grad()
def init_gpt_scion_(model: GPT, args):
    init_colnorm_(model.tok_emb.weight, radius=args.rho_embed, transpose=True)
    for block in model.blocks:
        init_spectral_(block.attn.q.weight, radius=args.rho_hidden)
        init_spectral_(block.attn.k.weight, radius=args.rho_hidden)
        init_spectral_(block.attn.v.weight, radius=args.rho_hidden)
        init_spectral_(block.attn.proj.weight, radius=args.rho_hidden)
        init_spectral_(block.mlp.gate.weight, radius=args.rho_hidden)
        init_spectral_(block.mlp.up.weight, radius=args.rho_hidden)
        init_spectral_(block.mlp.down.weight, radius=args.rho_hidden)
    init_sign_(model.lm_head.weight, radius=args.rho_out)


def resolve_group_lr(args, group: str) -> tuple[float, float]:
    peak = getattr(args, f"lr_{group}", None)
    if peak is None:
        peak = args.lr
    min_lr = getattr(args, f"min_lr_{group}", None)
    if min_lr is None:
        min_lr = args.min_lr
    if min_lr is None:
        min_lr = 0.1 * peak
    return peak, min_lr


def apply_auto_group_lrs(opt, summary: dict[str, dict[str, float]], args) -> str:
    parts = []
    for group in opt.param_groups:
        name = group.get("name", "group")
        l1 = summary.get(name, {}).get("l1")
        if l1 is None or not math.isfinite(l1) or l1 <= 0.0:
            continue
        old = group.get("auto_l1", l1)
        l1 = args.auto_lr_beta * old + (1.0 - args.auto_lr_beta) * l1
        group["auto_l1"] = l1

        lr = args.auto_lr_alpha / l1
        lr = min(max(lr, group.get("min_lr", 0.0)), group.get("max_lr", lr))
        group["peak_lr"] = lr
        parts.append(f"{name}={lr:.3e}")
    return "auto_lr " + ", ".join(parts) if parts else ""


@torch.no_grad()
def build_optimizer(model: GPT, args, device: torch.device):
    gns_dtype = torch.float16 if device.type == "cuda" else torch.float32
    skip = {id(model.tok_emb.weight), id(model.lm_head.weight)}
    hidden = [p for p in model.parameters() if p.requires_grad and id(p) not in skip]
    groups = []

    def hidden_lmo():
        if args.hidden_lmo == "gram-ns":
            return GramNewtonSchulzLMO(
                steps=args.pe_steps, work_dtype=gns_dtype
            )
        if args.hidden_lmo == "svd-filter":
            return HiddenSVDFilterLMO(
                steps=args.spi_steps,
                ridge=args.spi_ridge,
                refresh_interval=args.spi_refresh_interval,
                refresh_threshold=args.spi_refresh_threshold,
                iteration=args.spi_iteration,
                filter_ridge=args.filter_ridge,
            )
        return StreamingSVDSpectralLMO(
            steps=args.spi_steps,
            ridge=args.spi_ridge,
            refresh_interval=args.spi_refresh_interval,
            refresh_threshold=args.spi_refresh_threshold,
            iteration=args.spi_iteration,
        )

    def embed_lmo():
        if args.embed_lmo == "sign":
            return SignLMO()
        if args.embed_lmo == "rownorm":
            return RowNormLMO()
        return ColNormLMO(transpose=True)

    def out_lmo():
        if args.out_lmo == "colnorm":
            return ColNormLMO(transpose=True)
        if args.out_lmo == "rownorm":
            return RowNormLMO()
        return SignLMO()

    def add(name, params, dir_fn, radius):
        if not params:
            return
        peak_lr, min_lr = resolve_group_lr(args, name)
        radius = max(float(radius), 1e-12)
        group = {
            "name": name,
            "params": params,
            "dir_fn": dir_fn,
            "lr": peak_lr,
            "peak_lr": peak_lr,
            "max_lr": peak_lr,
            "min_lr": min_lr,
            "radius": radius,
            "weight_decay": 1.0 / radius,
        }
        groups.append(group)

    add(
        "embed",
        [model.tok_emb.weight],
        embed_lmo(),
        args.rho_embed,
    )
    add(
        "hidden",
        hidden,
        hidden_lmo(),
        args.rho_hidden,
    )
    add("out", [model.lm_head.weight], out_lmo(), args.rho_out)

    default_lr, _ = resolve_group_lr(args, "hidden")
    return ScionC(
        groups,
        lr=default_lr,
        beta2=args.beta2,
    )


def group_lmo(opt, name: str, cls):
    for group in opt.param_groups:
        if group.get("name") == name and isinstance(group.get("dir_fn"), cls):
            return group["dir_fn"]
    return None


def hidden_cov_lmo(opt):
    return group_lmo(opt, "hidden", HiddenSVDFilterLMO)


def register_hidden_cov_hooks(model: GPT, lmo):
    handles = []

    def covariance(x):
        flat = x.detach().reshape(-1, x.size(-1))
        return (flat.mT @ flat).float(), flat.size(0)

    def make_linear_hook(weight):
        def hook(_module, inputs):
            if not _module.training or not torch.is_grad_enabled():
                return
            cov, count = covariance(inputs[0])
            lmo.add_covariance(weight, cov, count)

        return hook

    def make_mlp_hook(module: MLP):
        def hook(_module, inputs):
            if not _module.training or not torch.is_grad_enabled():
                return
            cov, count = covariance(inputs[0])
            lmo.add_covariance(module.gate.weight, cov, count)
            lmo.add_covariance(module.up.weight, cov, count)

        return hook

    def make_qkv_hook(module: CausalSelfAttention):
        def hook(_module, inputs):
            if not _module.training or not torch.is_grad_enabled():
                return
            cov, count = covariance(inputs[0])
            lmo.add_covariance(module.q.weight, cov, count)
            lmo.add_covariance(module.k.weight, cov, count)
            lmo.add_covariance(module.v.weight, cov, count)

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
    init_gpt_scion_(raw_model, args)

    source = BatchSource(
        dataset.train, dataset.val, args.block_size, args.batch_size, device
    )
    opt = build_optimizer(raw_model, args, device)
    cov_lmo = hidden_cov_lmo(opt)
    if cov_lmo is not None:
        register_hidden_cov_hooks(raw_model, cov_lmo)
    conv_probe = (
        ConvergenceProbe(raw_model, opt, args)
        if args.track_convergence_stats or args.auto_lr_from_stats
        else None
    )
    if conv_probe is not None:
        conv_probe.register_hooks(raw_model)
        if args.compile:
            print("compile_disabled_for_convergence_stats")
            args.compile = False
    model, compile_seconds = maybe_compile(raw_model, source, args, amp_dtype, device)
    if cov_lmo is not None:
        cov_lmo.cov_accums.clear()
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
    effective_tokens = args.batch_size * args.block_size * args.grad_accum

    group_schedule = []
    for group in opt.param_groups:
        name = group.get("name", "group")
        peak_lr = group.get("peak_lr", group["lr"])
        min_lr = group.get("min_lr", args.min_lr)
        radius = group.get("radius", 1.0)
        wd = group.get("weight_decay", 0.0)
        group_schedule.append(
            f"{name}=({peak_lr:.3e}->{min_lr:.3e},rho={radius:g},wd={wd:.3g})"
        )
    print(
        "schedule "
        f"warmup_steps={warmup_steps} stable_steps={stable_steps} decay_steps={decay_steps} "
        f"optimizer=scionc prenorm={args.prenorm} dropout={args.dropout:.3f} "
        f"hidden_lmo={args.hidden_lmo} "
        f"embed_lmo={args.embed_lmo} out_lmo={args.out_lmo} "
        f"qkv=split spi_iteration={args.spi_iteration} "
        f"lr_groups=" + ", ".join(group_schedule)
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
        current_lrs = {}
        for group in opt.param_groups:
            peak_lr = group.get("peak_lr", args.lr)
            min_lr = group.get("min_lr", args.min_lr)
            group_lr = lr_at_step(
                step, args.max_iters, peak_lr, min_lr, warmup_steps, decay_steps
            )
            group["lr"] = group_lr
            current_lrs[group.get("name", f"group{len(current_lrs)}")] = group_lr
        lr = current_lrs.get("hidden", next(iter(current_lrs.values())))

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
                f"step {step:5d} | lr {lr:.3e} | train {train_loss:.4f} | val {val_loss:.4f} | "
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
            conv_text = conv_probe.capture(step, current_lrs)
            if conv_text:
                print(conv_text)
                if (
                    args.auto_lr_from_stats
                    and warmup_steps <= step < warmup_steps + stable_steps
                ):
                    auto_text = apply_auto_group_lrs(opt, conv_probe.summary, args)
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
        stats = getattr(group.get("dir_fn"), "stats", None)
        if stats:
            print(
                f"{group.get('name', 'group')}_lmo_stats "
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

    p.add_argument("--lr", type=float, default=3.5e-2)
    p.add_argument("--lr-embed", dest="lr_embed", type=float, default=None)
    p.add_argument("--lr-hidden", dest="lr_hidden", type=float, default=None)
    p.add_argument("--lr-out", "--lr-unembed", dest="lr_out", type=float, default=None)
    p.add_argument(
        "--min-lr",
        type=float,
        default=0.0,
        help="decay floor",
    )
    p.add_argument("--min-lr-embed", dest="min_lr_embed", type=float, default=None)
    p.add_argument("--min-lr-hidden", dest="min_lr_hidden", type=float, default=None)
    p.add_argument(
        "--min-lr-out", "--min-lr-unembed", dest="min_lr_out", type=float, default=None
    )
    p.add_argument("--beta2", type=float, default=0.93)
    p.add_argument(
        "--hidden-lmo",
        choices=["streaming-svd", "svd-filter", "gram-ns"],
        default="gram-ns",
        help="hidden-matrix LMO",
    )
    p.add_argument(
        "--embed-lmo",
        choices=["colnorm", "sign", "rownorm"],
        default="colnorm",
        help="embedding-table LMO",
    )
    p.add_argument(
        "--out-lmo",
        choices=["sign", "colnorm", "rownorm"],
        default="sign",
        help="output-head LMO",
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
    p.add_argument("--rho-embed", type=float, default=1.0)
    p.add_argument("--rho-hidden", type=float, default=3.0)
    p.add_argument("--rho-out", type=float, default=10.0)

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
        help="estimate same-batch LR aggressiveness with one extra forward",
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
        "--convergence-alpha",
        type=float,
        default=0.5,
        help="safety factor used when reporting LR predicted from L1 stats",
    )
    p.add_argument(
        "--convergence-probe",
        choices=["representative", "all"],
        default="representative",
        help="which parameters to include in convergence probes",
    )
    p.add_argument(
        "--auto-lr-from-stats",
        action="store_true",
        help="set group peak LRs from convergence L1 estimates after warmup",
    )
    p.add_argument(
        "--auto-lr-alpha",
        type=float,
        default=1.25,
        help="target normalized Lion-K step alpha = lr * L1",
    )
    p.add_argument(
        "--auto-lr-beta",
        type=float,
        default=0.8,
        help="EMA beta for L1 estimates used by --auto-lr-from-stats",
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
