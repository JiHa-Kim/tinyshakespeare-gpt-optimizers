import argparse
import json
import math
import time
from contextlib import nullcontext
from dataclasses import asdict, fields
from pathlib import Path

import torch

from scionc.compile_env import ensure_compile_env
from scionc.ulmos.core import (
    ColNormULMO,
    GramNewtonSchulzULMO,
    RowNormULMO,
    SignULMO,
    init_colnorm_,
    init_rownorm_,
    init_sign_,
    init_spectral_,
)
from scionc.ulmos.streaming_svd import HiddenSVDFilterULMO, StreamingSVDULMO
from scionc.optim.scion import ScionC
from scionc.optim.parametrization import (
    halving_factor,
    resolve_schedule,
    schedule_at_step,
    validate_step_scale,
)
from scionc.models.gpt import (
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

DEFAULT_RMS_RADII = {
    "embed": 0.70,
    "hidden": 0.051,
    "out": 0.022,
}
GROUP_NAMES = tuple(DEFAULT_RMS_RADII)
DEFAULT_COUNT_INCREMENT = 64 * 256
DEFAULT_MOMENTUM_RETENTION = 0.95
DEFAULT_BETA_HALF_LIFE = -DEFAULT_COUNT_INCREMENT / math.log2(
    DEFAULT_MOMENTUM_RETENTION
)
DEFAULT_SHRINKS = {
    "embed": 0.965,
    "hidden": 0.9883333333333333,
    "out": 0.9965,
}
DEFAULT_SHRINK_HALF_LIVES = {
    name: -DEFAULT_COUNT_INCREMENT / math.log2(shrink)
    for name, shrink in DEFAULT_SHRINKS.items()
}
DEFAULT_STEP_SCALE = 1.0
DEFAULT_BASE_ETA = 3.5e-2


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


def jsonable(value):
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    return str(value)


class MetricsLogger:
    def __init__(self, path: str, run_name: str = "") -> None:
        self.run_name = run_name
        self.file = None
        if path:
            metrics_path = Path(path)
            metrics_path.parent.mkdir(parents=True, exist_ok=True)
            self.file = metrics_path.open("a", encoding="utf-8", buffering=1)

    def write(self, event: str, **values) -> None:
        if self.file is None:
            return
        record = {"event": event, "run_name": self.run_name, **values}
        self.file.write(json.dumps(jsonable(record), separators=(",", ":")) + "\n")

    def close(self) -> None:
        if self.file is not None:
            self.file.close()
            self.file = None


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


def update_logit_stats(acc: dict[str, float], logits: torch.Tensor, targets: torch.Tensor) -> None:
    values = logits.detach().float()
    token_count = values.numel() // values.size(-1)
    centered = values - values.mean(dim=-1, keepdim=True)
    log_probs = torch.log_softmax(values, dim=-1)
    probs = log_probs.exp()
    top2 = torch.topk(values, k=min(2, values.size(-1)), dim=-1).values
    top_margin = (
        top2[..., 0] - top2[..., 1] if top2.size(-1) > 1 else torch.zeros_like(top2[..., 0])
    )
    target_probs = probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)

    acc["tokens"] += float(token_count)
    acc["logit_var"] += float(centered.square().mean(dim=-1).sum())
    acc["logit_margin"] += float(top_margin.sum())
    acc["softmax_entropy"] += float((-(probs * log_probs).sum(dim=-1)).sum())
    acc["softmax_max_prob"] += float(probs.max(dim=-1).values.sum())
    acc["target_prob"] += float(target_probs.sum())


def finalize_logit_stats(acc: dict[str, float]) -> dict[str, float]:
    tokens = max(acc.get("tokens", 0.0), 1.0)
    return {
        "logit_std": math.sqrt(max(0.0, acc["logit_var"] / tokens)),
        "logit_margin": acc["logit_margin"] / tokens,
        "softmax_entropy": acc["softmax_entropy"] / tokens,
        "softmax_max_prob": acc["softmax_max_prob"] / tokens,
        "target_prob": acc["target_prob"] / tokens,
    }


@torch.inference_mode()
def estimate_val_metrics(
    model: GPT,
    source: BatchSource,
    eval_iters: int,
    amp_dtype: torch.dtype | None,
    collect_logit_stats: bool = False,
) -> tuple[float, dict[str, float]]:
    was_training = model.training
    model.eval()
    total = 0.0
    logit_acc = {
        "tokens": 0.0,
        "logit_var": 0.0,
        "logit_margin": 0.0,
        "softmax_entropy": 0.0,
        "softmax_max_prob": 0.0,
        "target_prob": 0.0,
    }
    with amp_ctx(amp_dtype):
        for _ in range(eval_iters):
            xb, yb = source.get("val")
            logits, loss = model(xb, yb)
            total += float(loss)
            if collect_logit_stats:
                update_logit_stats(logit_acc, logits, yb)
    model.train(was_training)
    stats = finalize_logit_stats(logit_acc) if collect_logit_stats else {}
    return total / eval_iters, stats


def scale_from_coordinate(
    linear: float | None, log2_value: float | None, label: str
) -> float | None:
    if linear is not None and log2_value is not None:
        raise ValueError(f"set either {label} or log2 {label}, not both")
    if log2_value is None:
        return linear
    if not math.isfinite(log2_value):
        raise ValueError(f"invalid log2 {label}: {log2_value}")
    try:
        return 2.0**log2_value
    except OverflowError as exc:
        raise ValueError(f"invalid log2 {label}: {log2_value}") from exc


def resolve_group_step_scale(args, group: str) -> tuple[float, float]:
    peak = scale_from_coordinate(
        getattr(args, f"step_scale_{group}", None),
        getattr(args, f"log2_step_scale_{group}", None),
        f"{group} step scale",
    )
    if peak is None:
        peak = scale_from_coordinate(
            args.step_scale,
            args.log2_step_scale,
            "global step scale",
        )
    if peak is None:
        peak = DEFAULT_STEP_SCALE
    floor = getattr(args, f"min_step_scale_{group}", None)
    if floor is None:
        floor = args.min_step_scale
    if floor is None:
        floor = 0.1 * peak
    peak = validate_step_scale(peak, f"{group} step scale")
    floor = validate_step_scale(floor, f"{group} minimum step scale")
    if floor > peak:
        raise ValueError(
            f"invalid {group} minimum step scale: {floor}; expected <= peak {peak}"
        )
    return peak, floor


def resolve_group_rms_radius(args, group: str) -> float:
    radius = getattr(args, f"rms_radius_{group}", None)
    if radius is None:
        radius = getattr(args, "rms_radius", None)
    if radius is None:
        radius = DEFAULT_RMS_RADII[group]
    if radius <= 0.0:
        raise ValueError(f"invalid {group} target RMS radius: {radius}")
    return float(radius)


def count_increment(args) -> int:
    return args.batch_size * args.block_size * args.grad_accum


def resolve_group_shrink_half_life(args, group: str) -> float:
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

        step_scale = args.auto_action_scale / (l1 * float(group["base_eta"]))
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


def input_output_tied(model: GPT) -> bool:
    return model.tok_emb.weight is model.lm_head.weight


def make_edge_ulmo(kind: str):
    if kind == "colnorm":
        return ColNormULMO(transpose=True)
    if kind == "rownorm":
        return RowNormULMO()
    if kind == "sign":
        return SignULMO()
    raise ValueError(f"unsupported edge ULMO: {kind}")


def hidden_params(model: GPT) -> list[torch.Tensor]:
    skip = {id(model.tok_emb.weight), id(model.lm_head.weight)}
    return [p for p in model.parameters() if p.requires_grad and id(p) not in skip]


def group_schedule_ratio(group: dict, scheduled_step_scale: float) -> float:
    peak_step_scale = float(group["peak_step_scale"])
    if peak_step_scale == 0.0:
        return 0.0
    ratio = scheduled_step_scale / peak_step_scale
    if not math.isfinite(ratio) or ratio < 0.0:
        raise ValueError(
            f"invalid schedule ratio for {group.get('name', 'group')}: {ratio}"
        )
    return ratio


def group_action(group: dict, scheduled_step_scale: float) -> tuple[float, float]:
    peak_scale = float(group["peak_step_scale"])
    validate_step_scale(scheduled_step_scale, f"{group.get('name', 'group')} schedule")
    if peak_scale > 0.0 and scheduled_step_scale > peak_scale * (1.0 + 1e-12):
        raise ValueError(
            f"invalid {group.get('name', 'group')} scheduled scale "
            f"{scheduled_step_scale}; expected <= peak {peak_scale}"
        )

    ratio = group_schedule_ratio(group, scheduled_step_scale)
    peak_shrink = float(group["peak_shrink"])
    shrink = (
        peak_shrink
        if group.get("shrink_schedule") == "constant"
        else peak_shrink**ratio
    )
    return shrink, float(group["base_eta"]) * scheduled_step_scale


def _view_shape(p: torch.Tensor, transpose: bool = False) -> tuple[int, int]:
    if p.ndim != 2:
        return p.numel(), 1
    rows, cols = p.shape
    return (cols, rows) if transpose else (rows, cols)


def spectral_atom_sq(p: torch.Tensor, input_like: bool = False) -> float:
    rows, cols = _view_shape(p)
    if rows <= 0 or cols <= 0:
        return 0.0
    scale_sq = rows / cols
    if input_like:
        scale_sq = max(1.0, scale_sq)
    return min(rows, cols) * scale_sq / (rows * cols)


def ulmo_atom_sq(p: torch.Tensor, ulmo) -> float:
    if isinstance(ulmo, ColNormULMO):
        return 1.0
    if isinstance(ulmo, (RowNormULMO, SignULMO)):
        _, cols = _view_shape(p, getattr(ulmo, "transpose", False))
        return 1.0 / (cols * cols)
    if isinstance(ulmo, (GramNewtonSchulzULMO, StreamingSVDULMO)):
        return spectral_atom_sq(p, getattr(ulmo, "input_like", False))
    return 1.0 / max(p.numel(), 1)


@torch.no_grad()
def current_group_rms(group: dict) -> float:
    total = sum(p.numel() for p in group["params"])
    if total <= 0:
        return 0.0
    sq = sum(float(p.detach().float().square().sum()) for p in group["params"])
    return math.sqrt(sq / total)


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


def init_radius_for_weight_rms(p: torch.Tensor, ulmo, rms_radius: float) -> float:
    atom_sq = ulmo_atom_sq(p, ulmo)
    if atom_sq <= 0.0:
        return rms_radius
    return rms_radius / math.sqrt(atom_sq)


@torch.no_grad()
def init_from_actions_(groups: list[dict]) -> None:
    for group in groups:
        ulmo = group["ulmo"]
        for p in group["params"]:
            radius = init_radius_for_weight_rms(p, ulmo, float(group["rms_radius"]))
            init_like_ulmo_(p, ulmo, radius)


def action_group_fields(
    name: str, args, delta_tau: int, memory_beta: float
) -> dict:
    peak_step_scale, min_step_scale = resolve_group_step_scale(args, name)
    shrink_half_life = resolve_group_shrink_half_life(args, name)
    peak_shrink = halving_factor(
        delta_tau,
        shrink_half_life,
        f"{name}_shrink_half_life",
    )
    rms_radius = resolve_group_rms_radius(args, name)
    fields = {
        "rms_radius": rms_radius,
        "memory_beta": memory_beta,
        "base_eta": DEFAULT_BASE_ETA,
        "peak_eta": DEFAULT_BASE_ETA * peak_step_scale,
        "peak_step_scale": peak_step_scale,
        "max_step_scale": peak_step_scale,
        "min_step_scale": min_step_scale,
        "peak_shrink": peak_shrink,
        "shrink_half_life": shrink_half_life,
        "shrink_schedule": args.shrink_schedule,
        "rms_solve": args.rms_solve,
        "beta_half_life": args.beta_half_life,
        "count_increment": delta_tau,
    }
    shrink, eta = group_action(fields, peak_step_scale)
    fields.update(
        step_scale=peak_step_scale,
        shrink=shrink,
        lr=eta,
    )
    return fields


def optimizer_group(
    name: str,
    params: list[torch.Tensor],
    ulmo,
    args,
    delta_tau: int,
    memory_beta: float,
) -> dict | None:
    if not params:
        return None
    return {
        "name": name,
        "params": params,
        "ulmo": ulmo,
        **action_group_fields(name, args, delta_tau, memory_beta),
    }


def optimizer_group_specs(model: GPT, args, work_dtype: torch.dtype):
    tied = input_output_tied(model)
    specs = [
        (
            "embed",
            [model.tok_emb.weight],
            SignULMO() if tied else make_edge_ulmo(args.embed_ulmo),
        ),
        (
            "hidden",
            hidden_params(model),
            make_hidden_ulmo(args, work_dtype),
        ),
    ]
    if not tied:
        specs.append(("out", [model.lm_head.weight], make_edge_ulmo(args.out_ulmo)))
    return specs


@torch.no_grad()
def build_optimizer(model: GPT, args, device: torch.device):
    delta_tau = count_increment(args)
    memory_beta = halving_factor(delta_tau, args.beta_half_life, "beta_half_life")
    work_dtype = torch.float16 if device.type == "cuda" else torch.float32
    groups = []
    for name, params, ulmo in optimizer_group_specs(model, args, work_dtype):
        group = optimizer_group(name, params, ulmo, args, delta_tau, memory_beta)
        if group is not None:
            groups.append(group)
    init_from_actions_(groups)

    hidden_group = next(group for group in groups if group["name"] == "hidden")
    return ScionC(
        groups,
        lr=hidden_group["lr"],
        readout_mu=args.readout_mu,
        memory_beta=memory_beta,
    )


def hidden_cov_ulmo(opt):
    for group in opt.param_groups:
        ulmo = group.get("ulmo")
        if group.get("name") == "hidden" and isinstance(ulmo, HiddenSVDFilterULMO):
            return ulmo
    return None


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


def load_torch_checkpoint(path: Path, device: torch.device):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def load_checkpoint(path: Path, device: torch.device):
    ckpt = load_torch_checkpoint(path, device)
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
    ensure_compile_env()
    model = torch.compile(model)
    xb, yb = source.get("train")
    t0 = sync_now(device)
    model.zero_grad(set_to_none=True)
    with amp_ctx(amp_dtype):
        _, loss = model(xb, yb)
    loss.backward()
    model.zero_grad(set_to_none=True)
    return model, sync_now(device) - t0


def configure_runtime(args) -> tuple[torch.device, torch.dtype | None]:
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
    return device, amp_dtype


def load_dataset(args) -> CharDataset:
    data_path = Path(args.data_path)
    maybe_download_tiny_shakespeare(data_path)
    return CharDataset(data_path)


def build_model(args, dataset: CharDataset, device: torch.device) -> GPT:
    cfg = GPTConfig(
        vocab_size=len(dataset.chars),
        block_size=args.block_size,
        n_layer=args.n_layer,
        n_head=args.n_head,
        d_model=args.d_model,
        rope_base=args.rope_base,
        prenorm=args.prenorm,
        dropout=args.dropout,
        tie_weights=args.tie_weights,
    )
    return GPT(cfg).to(device)


def resolve_training_schedule(args) -> tuple[int, int, int]:
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
    return resolve_schedule(args.max_iters, warmup_steps, decay_steps)


def format_optimizer_schedule(opt) -> str:
    parts = []
    for group in opt.param_groups:
        name = group.get("name", "group")
        peak_step_scale = group.get("peak_step_scale", group["step_scale"])
        min_step_scale = group.get("min_step_scale", 0.0)
        peak_shrink, peak_eta = group_action(group, peak_step_scale)
        min_shrink, min_eta = group_action(group, min_step_scale)
        rms_radius = group.get("rms_radius", math.nan)
        shrink_half_life = group.get("shrink_half_life", math.inf)
        parts.append(
            f"{name}=(rw={rms_radius:g},"
            f"s={peak_step_scale:.3g}->{min_step_scale:.3g},"
            f"eta={peak_eta:.3e}->{min_eta:.3e},"
            f"h_shrink={shrink_half_life:.3g},"
            f"shrink_sched={group.get('shrink_schedule', 'scheduled')},"
            f"shrink={peak_shrink:.6f}->{min_shrink:.6f})"
        )
    return ", ".join(parts)


@torch.no_grad()
def optimizer_rms_state(opt) -> dict[str, dict[str, float]]:
    out = {}
    for group in opt.param_groups:
        target = float(group.get("rms_radius", math.nan))
        current = current_group_rms(group)
        out[group.get("name", f"group{len(out)}")] = {
            "param_rms": current,
            "target_rms": target,
            "rms_ratio": current / target if target and target > 0.0 else math.nan,
        }
    return out


def rms_state_text(state: dict[str, dict[str, float]]) -> str:
    if not state:
        return ""
    parts = []
    for name, values in state.items():
        parts.append(
            f"{name}={values['param_rms']:.3g}/{values['target_rms']:.3g}"
        )
    return " | weight_rms " + "; ".join(parts)


def apply_scheduled_etas(
    opt, step: int, max_steps: int, warmup_steps: int, decay_steps: int
) -> dict[str, float]:
    current_etas = {}
    for group in opt.param_groups:
        peak_step_scale = group.get("peak_step_scale", group["step_scale"])
        min_step_scale = group.get("min_step_scale", 0.0)
        step_scale = schedule_at_step(
            step,
            max_steps,
            peak_step_scale,
            min_step_scale,
            warmup_steps,
            decay_steps,
        )
        shrink, eta = group_action(group, step_scale)
        group["step_scale"] = step_scale
        group["shrink"] = shrink
        group["lr"] = eta
        current_etas[group.get("name", f"group{len(current_etas)}")] = eta
    return current_etas


def line_probe_active(args, step: int) -> bool:
    return (
        args.track_line_probe
        and args.grad_accum == 1
        and args.line_probe_interval > 0
        and step % args.line_probe_interval == 0
    )


def run_line_probe(
    model,
    step: int,
    batch,
    rng_before,
    loss_before: float | None,
    params_before,
    curve_scales: list[float],
    line_stats: dict[str, dict],
    amp_dtype: torch.dtype | None,
    device: torch.device,
) -> None:
    if batch is None or rng_before is None or loss_before is None:
        return

    rng_after = capture_rng(device)
    if params_before is None:
        restore_rng(rng_before, device)
        with torch.no_grad(), amp_ctx(amp_dtype):
            _, loss_after = model(*batch)
        restore_rng(rng_after, device)
        loss_after_value = float(loss_after.detach())
    else:
        snapshot = finish_line_snapshot(params_before)
        curve_losses = []
        for scale in curve_scales:
            apply_line_scale(snapshot, scale)
            restore_rng(rng_before, device)
            with torch.no_grad(), amp_ctx(amp_dtype):
                _, curve_loss = model(*batch)
            curve_losses.append((scale, float(curve_loss.detach())))
        apply_line_scale(snapshot, 1.0)
        restore_rng(rng_after, device)
        loss_after_value = min(curve_losses, key=lambda item: abs(item[0] - 1.0))[1]
        curve_text = line_curve_text(step, curve_losses)
        if curve_text:
            print(curve_text)

    line_text = line_probe_text(step, loss_before, loss_after_value, line_stats)
    if line_text:
        print(line_text)


def train(args):
    device, amp_dtype = configure_runtime(args)
    metrics = MetricsLogger(args.metrics_jsonl, args.run_name)
    line_curve_scales = parse_line_scales(args.line_curve_scales)
    if line_curve_scales:
        args.track_line_probe = True

    dataset = load_dataset(args)
    raw_model = build_model(args, dataset, device)
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

    warmup_steps, stable_steps, decay_steps = resolve_training_schedule(args)
    effective_tokens = count_increment(args)
    first_group = opt.param_groups[0]
    readout_mu = first_group.get("readout_mu", args.readout_mu)
    memory_beta = first_group.get("memory_beta", math.nan)
    beta_half_life = first_group.get("beta_half_life", math.nan)
    io_weights = "tied" if input_output_tied(raw_model) else "untied"

    eta_groups = format_optimizer_schedule(opt)
    print(
        "schedule "
        f"warmup_steps={warmup_steps} stable_steps={stable_steps} decay_steps={decay_steps} "
        f"count_increment={effective_tokens} "
        f"beta_half_life={beta_half_life:.3g} beta={memory_beta:.6f} "
        f"readout_mu={readout_mu:.3g} "
        f"optimizer=scionc prenorm={args.prenorm} dropout={args.dropout:.3f} "
        f"hidden_ulmo={args.hidden_ulmo} "
        f"io_weights={io_weights} embed_ulmo={args.embed_ulmo} out_ulmo={args.out_ulmo} "
        f"qkv=split spi_iteration={args.spi_iteration} "
        f"rms_solve={args.rms_solve} "
        f"eta_groups={eta_groups}"
    )
    metrics.write(
        "config",
        args=vars(args),
        schedule={
            "warmup_steps": warmup_steps,
            "stable_steps": stable_steps,
            "decay_steps": decay_steps,
            "count_increment": effective_tokens,
        },
        optimizer={
            "name": "scionc",
            "beta_half_life": beta_half_life,
            "beta": memory_beta,
            "readout_mu": readout_mu,
            "rms_solve": args.rms_solve,
            "eta_groups": eta_groups,
        },
        model={
            "prenorm": args.prenorm,
            "dropout": args.dropout,
            "hidden_ulmo": args.hidden_ulmo,
            "io_weights": io_weights,
            "embed_ulmo": args.embed_ulmo,
            "out_ulmo": args.out_ulmo,
            "qkv": "split",
            "spi_iteration": args.spi_iteration,
        },
    )
    if args.track_line_probe and args.grad_accum != 1:
        print("line_probe_disabled_requires_grad_accum_1")
    if line_curve_scales:
        print("line_curve_scales " + ",".join(f"{x:g}" for x in line_curve_scales))

    total_opt_steps = 0
    best_val = float("inf")
    max_val = float("-inf")
    last_train_loss = float("nan")
    last_val_loss = float("nan")
    initial_val = None
    diverged = False
    diverge_reason = ""
    train_start = sync_now(device)
    step_stat_accum = {}

    for step in range(args.max_iters):
        current_etas = apply_scheduled_etas(
            opt, step, args.max_iters, warmup_steps, decay_steps
        )
        eta = current_etas.get("hidden", next(iter(current_etas.values())))

        if step % args.eval_interval == 0 or step == args.max_iters - 1:
            train_loss = float(last_train_loss)
            val_loss, logit_stats = estimate_val_metrics(
                model,
                source,
                args.eval_iters,
                amp_dtype,
                args.track_logit_stats,
            )
            last_val_loss = val_loss
            opt_stats = (
                consume_step_stats(step_stat_accum) if args.track_step_stats else {}
            )
            weight_rms = optimizer_rms_state(opt)

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
                        path = Path(args.out_path)
                        save_checkpoint(
                            path.with_name(f"{path.stem}_final{path.suffix}"),
                            raw_model,
                            dataset,
                            args,
                        )

            elapsed = max(sync_now(device) - train_start, 1e-9)
            mem_text = cuda_memory_text(device)
            opt_text = step_stats_text(opt_stats)
            rms_text = rms_state_text(weight_rms)
            logit_text = (
                " | logits "
                f"std={logit_stats['logit_std']:.3f},"
                f"H={logit_stats['softmax_entropy']:.3f},"
                f"pmax={logit_stats['softmax_max_prob']:.3f}"
                if logit_stats
                else ""
            )
            print(
                f"step {step:5d} | eta {eta:.3e} | train {train_loss:.4f} | val {val_loss:.4f} | "
                f"best_val {best_val:.4f} | train_seconds {elapsed:.3f} | "
                f"tok/s {(total_opt_steps * effective_tokens) / elapsed:.0f}"
                f"{mem_text}{logit_text}{rms_text}{opt_text}"
            )
            memory = cuda_memory_stats(device)
            metrics.write(
                "eval",
                step=step,
                total_opt_steps=total_opt_steps,
                eta=eta,
                etas=current_etas,
                train_loss=train_loss,
                val_loss=val_loss,
                best_val=best_val,
                max_val=max_val,
                train_seconds=elapsed,
                tokens_per_second=(total_opt_steps * effective_tokens) / elapsed,
                cuda_memory=memory,
                logit_stats=logit_stats,
                weight_rms=weight_rms,
                step_stats=opt_stats,
            )
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

        line_active = line_probe_active(args, step)
        line_batch = None
        line_rng_before = None
        line_loss_before = None
        opt.zero_grad(set_to_none=True)
        if conv_probe is not None:
            conv_probe.start_step(step)
        train_loss = None
        for micro_step in range(args.grad_accum):
            batch = source.get("train")
            if line_active and micro_step == 0:
                line_batch = batch
                line_rng_before = capture_rng(device)
            with amp_ctx(amp_dtype):
                _, loss = model(*batch)
                loss = loss / args.grad_accum
            loss_value = loss.detach()
            if line_active and micro_step == 0:
                line_loss_before = float(loss_value)
            train_loss = loss_value if train_loss is None else train_loss + loss_value
            loss.backward()
        if diverged:
            print(f"diverged {diverge_reason}")
            break
        last_train_loss = train_loss if train_loss is not None else float("nan")

        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(raw_model.parameters(), args.grad_clip)
        if conv_probe is not None:
            conv_text = conv_probe.capture(step, current_etas)
            if conv_text:
                print(conv_text)
                metrics.write(
                    "convergence",
                    step=step,
                    total_opt_steps=total_opt_steps,
                    etas=current_etas,
                    groups=conv_probe.summary,
                )
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
        if line_active:
            run_line_probe(
                model,
                step,
                line_batch,
                line_rng_before,
                line_loss_before,
                line_params_before,
                line_curve_scales,
                line_stats,
                amp_dtype,
                device,
            )

    if not (args.skip_sample or diverged):
        prompt = args.prompt or "\n"
        x = torch.tensor([dataset.encode(prompt)], dtype=torch.long, device=device)
        texts = generate_texts(
            raw_model,
            x,
            dataset.decode,
            args.sample_count,
            args.sample_tokens,
            args.temperature,
            args.top_k,
        )
        if not write_sample_report(args, texts):
            print_samples(texts)

    for group in opt.param_groups:
        stats = getattr(group.get("ulmo"), "stats", None)
        if stats:
            print(
                f"{group.get('name', 'group')}_ulmo_stats "
                + " ".join(f"{k}={v}" for k, v in stats.items())
            )

    last_train_loss = float(last_train_loss)
    result = {
        "best_val": best_val,
        "final_train": last_train_loss,
        "final_val": last_val_loss,
        "compile_seconds": compile_seconds,
        "initial_val": float("nan") if initial_val is None else initial_val,
        "max_val": max_val,
        "diverged": diverged,
        "diverge_reason": diverge_reason,
        "warmup_steps": warmup_steps,
        "stable_steps": stable_steps,
        "decay_steps": decay_steps,
    }
    metrics.write("final", **result)
    metrics.close()
    return result


@torch.inference_mode()
def sample(args):
    device, _ = configure_runtime(args)
    model, stoi, itos = load_checkpoint(Path(args.out_path), device)
    prompt = args.prompt or "\n"
    bad = [c for c in prompt if c not in stoi]
    if bad:
        raise ValueError(f"prompt contains unseen chars: {bad}")
    x = torch.tensor([[stoi[c] for c in prompt]], dtype=torch.long, device=device)
    texts = generate_texts(
        model,
        x,
        lambda ids: "".join(itos[int(i)] for i in ids),
        args.sample_count,
        args.sample_tokens,
        args.temperature,
        args.top_k,
    )

    if write_sample_report(args, texts):
        return

    print_samples(texts)


def generate_texts(
    model,
    x: torch.Tensor,
    decode,
    sample_count: int,
    sample_tokens: int,
    temperature: float,
    top_k: int,
) -> list[str]:
    texts = []
    for _ in range(sample_count):
        y = model.generate(
            x,
            max_new_tokens=sample_tokens,
            temperature=temperature,
            top_k=top_k,
        )
        texts.append(decode(y[0].tolist()))
    return texts


def write_sample_report(args, texts: list[str]) -> bool:
    if not args.sample_out:
        return False
    path = Path(args.sample_out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(sample_report(args, texts), encoding="utf-8")
    print(f"wrote_samples {path}")
    return True


def print_samples(texts: list[str]) -> None:
    for i, text in enumerate(texts, start=1):
        if len(texts) > 1:
            print(f"\n--- sample {i} ---\n")
        elif i == 1:
            print("\n--- sample ---\n")
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
    device, amp_dtype = configure_runtime(args)
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
    p.add_argument(
        "--tie-weights",
        action="store_true",
        help="share input embedding and output head weights",
    )

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
        default=None,
        help="linear peak eta multiplier; peak eta is 0.035 times this value",
    )
    p.add_argument(
        "--log2-step-scale",
        type=float,
        default=None,
        help="base-2 log of the peak eta multiplier",
    )
    for group in GROUP_NAMES:
        p.add_argument(f"--step-scale-{group}", type=float, default=None)
        p.add_argument(f"--log2-step-scale-{group}", type=float, default=None)
    p.add_argument(
        "--min-step-scale",
        type=float,
        default=0.0,
        help="decay floor for the eta multiplier; must be <= peak",
    )
    for group in GROUP_NAMES:
        p.add_argument(f"--min-step-scale-{group}", type=float, default=None)
    p.add_argument(
        "--rms-radius",
        dest="rms_radius",
        type=float,
        default=None,
        help=(
            "peak actual entrywise weight RMS target for all optimizer groups; "
            "defaults are embed=0.70, hidden=0.051, out=0.022"
        ),
    )
    for group in GROUP_NAMES:
        p.add_argument(f"--rms-radius-{group}", type=float, default=None)
    p.add_argument(
        "--beta-half-life",
        type=float,
        default=DEFAULT_BETA_HALF_LIFE,
        help="momentum-state retention half-life in processed tokens",
    )
    p.add_argument(
        "--readout-mu",
        type=float,
        default=1.0,
        help="ULMO readout blend between current gradient and momentum state",
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
        help="embedding-table ULMO; tied weights force Sign",
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
        help="weight shrink half-life in processed tokens for all groups",
    )
    for group in GROUP_NAMES:
        p.add_argument(f"--shrink-half-life-{group}", type=float, default=None)
    p.add_argument(
        "--shrink-schedule",
        choices=["scheduled", "constant"],
        default="scheduled",
        help=(
            "scheduled applies the WSD ratio to the shrink halving exponent; "
            "constant keeps shrink at its peak half-life"
        ),
    )
    p.add_argument(
        "--rms-solve",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="solve eta per step to target rms-radius, capped by the eta schedule",
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
        "--track-logit-stats",
        action="store_true",
        help="log cheap validation-batch softmax/logit statistics on eval lines",
    )
    p.add_argument(
        "--metrics-jsonl",
        default="",
        help="append structured config/eval/convergence/final records to this JSONL path",
    )
    p.add_argument(
        "--run-name",
        default="",
        help="optional run label included in structured metrics records",
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
        "--convergence-support-steps",
        type=int,
        default=7,
        help="Gram-NS polar-support steps for spectral dual stats",
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
