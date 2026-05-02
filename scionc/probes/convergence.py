import math
from copy import copy
from dataclasses import dataclass

import torch

from scionc.ulmos import (
    ColNormULMO,
    GramNewtonSchulzULMO,
    RowNormULMO,
    SignULMO,
    StreamingSVDULMO,
    gram_newton_schulz_uvt,
)
from scionc.models import GPT


@dataclass
class ConvergenceItem:
    name: str
    group: str
    param: torch.Tensor
    ulmo: object


def median(values: list[float]) -> float:
    if not values:
        return float("nan")
    values = sorted(values)
    mid = len(values) // 2
    if len(values) % 2:
        return values[mid]
    return 0.5 * (values[mid - 1] + values[mid])


def oriented_matrix(x: torch.Tensor, ulmo) -> torch.Tensor:
    return x.mT if getattr(ulmo, "transpose", False) else x


def spectral_ulmo_scale(x: torch.Tensor, ulmo) -> float:
    scale = math.sqrt(x.size(0) / x.size(1))
    return max(1.0, scale) if getattr(ulmo, "input_like", False) else scale


def is_spectral_ulmo(ulmo) -> bool:
    return isinstance(ulmo, GramNewtonSchulzULMO | StreamingSVDULMO)


def spectral_support_dual(
    x: torch.Tensor, ulmo, param: torch.Tensor | None, eps: float
) -> float:
    x = x.detach()
    if isinstance(ulmo, GramNewtonSchulzULMO):
        x32 = x.float()
        polar = gram_newton_schulz_uvt(
            x32, ulmo.steps, ulmo.eps, ulmo.work_dtype, ulmo.bound_safety
        ).float()
        return max(0.0, float((x32 * polar).sum()) * spectral_ulmo_scale(x32, ulmo))

    stat_ulmo = copy(ulmo)
    stat_ulmo.states = dict(ulmo.states)
    stat_ulmo.stats = dict(ulmo.stats)
    stat_ulmo._param_key = id(param) if param is not None else None
    update = stat_ulmo(x.clone(memory_format=torch.preserve_format))
    return max(0.0, float(-(x.float() * update.float()).sum()))


def dual_norm(
    x: torch.Tensor, ulmo, eps: float = 1e-12, param: torch.Tensor | None = None
) -> float:
    x = x.float()
    if is_spectral_ulmo(ulmo):
        return spectral_support_dual(x, ulmo, param, eps)

    y = oriented_matrix(x, ulmo)
    if isinstance(ulmo, SignULMO):
        return float(y.abs().sum() / max(y.size(1), 1))
    if isinstance(ulmo, ColNormULMO):
        return float(
            torch.linalg.vector_norm(y, dim=0).sum() * math.sqrt(y.size(0))
        )
    if isinstance(ulmo, RowNormULMO):
        return float(
            torch.linalg.vector_norm(y, dim=1).sum() / math.sqrt(max(y.size(1), 1))
        )
    return float(torch.linalg.vector_norm(x).clamp_min(eps))


def spectral_norm_power(x: torch.Tensor, eps: float = 1e-12, steps: int = 4) -> float:
    x = x.float()
    if x.ndim != 2 or x.numel() == 0:
        return float(torch.linalg.vector_norm(x).clamp_min(eps))
    v = torch.ones(x.size(1), dtype=x.dtype, device=x.device)
    v = v / torch.linalg.vector_norm(v).clamp_min(eps)
    for _ in range(steps):
        u = x @ v
        u_norm = torch.linalg.vector_norm(u)
        if float(u_norm) <= eps:
            return 0.0
        u = u / u_norm
        v = x.mT @ u
        v_norm = torch.linalg.vector_norm(v)
        if float(v_norm) <= eps:
            return 0.0
        v = v / v_norm
    return float(torch.linalg.vector_norm(x @ v).clamp_min(eps))


def primal_norm(x: torch.Tensor, ulmo, eps: float = 1e-12) -> float:
    x = x.float()
    if is_spectral_ulmo(ulmo):
        return spectral_norm_power(x, eps) / spectral_ulmo_scale(x, ulmo)

    y = oriented_matrix(x, ulmo)
    if isinstance(ulmo, SignULMO):
        return float(y.abs().max() * max(y.size(1), 1))
    if isinstance(ulmo, ColNormULMO):
        return float(
            torch.linalg.vector_norm(y, dim=0).max() / math.sqrt(y.size(0))
        )
    if isinstance(ulmo, RowNormULMO):
        return float(
            torch.linalg.vector_norm(y, dim=1).max() * math.sqrt(max(y.size(1), 1))
        )
    return float(torch.linalg.vector_norm(x).clamp_min(eps))


def nuclear_rank(x: torch.Tensor, ulmo, eps: float = 1e-12) -> float:
    if not is_spectral_ulmo(ulmo):
        return float("nan")
    x = x.float()
    fro_sq = x.square().sum().clamp_min(eps)
    nuc = dual_norm(x, ulmo, eps) / max(spectral_ulmo_scale(x, ulmo), eps)
    return float((nuc * nuc) / fro_sq)


def stable_rank_from_input(x: torch.Tensor, eps: float = 1e-12) -> float:
    flat = x.detach().reshape(-1, x.size(-1)).float()
    gram = (flat.mT @ flat).float()
    fro_sq = gram.diagonal().sum().clamp_min(eps)
    op_sq = torch.linalg.eigvalsh(gram).amax().clamp_min(eps)
    return float(fro_sq / op_sq)


class ConvergenceProbe:
    def __init__(self, model: GPT, opt, args):
        self.interval = args.convergence_interval
        self.action_scale = args.convergence_action_scale
        self.eps = 1e-12
        self.active = False
        self.prev: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
        self.input_sr: dict[int, float] = {}
        self.summary: dict[str, dict[str, float]] = {}
        self.items = self._items(model, opt, args.convergence_probe)

    def _items(self, model: GPT, opt, probe: str) -> list[ConvergenceItem]:
        groups = {
            id(p): (
                group.get("name", "group"),
                group["ulmo"],
            )
            for group in opt.param_groups
            for p in group["params"]
        }
        keep = self._probe_names(model) if probe == "representative" else None
        items = []
        for name, p in model.named_parameters():
            if not p.requires_grad or id(p) not in groups:
                continue
            if keep is not None and name not in keep:
                continue
            group, ulmo = groups[id(p)]
            items.append(ConvergenceItem(name, group, p, ulmo))
        return items

    def _probe_names(self, model: GPT) -> set[str]:
        names = {"tok_emb.weight", "lm_head.weight"}
        block_ids = sorted({0, len(model.blocks) // 2, len(model.blocks) - 1})
        suffixes = (
            "attn.q.weight",
            "attn.k.weight",
            "attn.v.weight",
            "attn.proj.weight",
            "mlp.gate.weight",
            "mlp.up.weight",
            "mlp.down.weight",
        )
        for block_id in block_ids:
            names.update(f"blocks.{block_id}.{suffix}" for suffix in suffixes)
        return names

    def start_step(self, step: int) -> None:
        self.active = self.interval > 0 and step % self.interval == 0
        if self.active:
            self.input_sr.clear()

    def register_hooks(self, model: GPT):
        selected = {id(item.param) for item in self.items}
        handles = []
        for module in model.modules():
            if not isinstance(module, torch.nn.Linear):
                continue
            weight = getattr(module, "weight", None)
            if weight is None or id(weight) not in selected:
                continue
            handles.append(module.register_forward_pre_hook(self._make_hook(weight)))
        return handles

    def _make_hook(self, weight: torch.Tensor):
        def hook(module, inputs):
            if not (self.active and module.training and torch.is_grad_enabled()):
                return
            with torch.no_grad():
                self.input_sr[id(weight)] = stable_rank_from_input(inputs[0], self.eps)

        return hook

    def capture(self, step: int, current_etas: dict[str, float]) -> str:
        report = self.active
        if not report:
            self.summary = {}
        grouped: dict[str, dict[str, list[float]]] = {}
        for item in self.items:
            grad = item.param.grad
            if grad is None:
                continue
            current_grad = grad.detach().float().cpu()
            current_param = item.param.detach().float().cpu()
            previous = self.prev.get(id(item.param))

            if report:
                stats = grouped.setdefault(item.group, {})
                grad_dual = dual_norm(current_grad, item.ulmo, self.eps, item.param)
                self._append(stats, "gdual", grad_dual)
                self._append(stats, "eta", current_etas.get(item.group, float("nan")))

                if previous is not None:
                    prev_grad, prev_param = previous
                    dgrad = dual_norm(
                        current_grad - prev_grad, item.ulmo, self.eps, item.param
                    )
                    dparam = primal_norm(
                        current_param - prev_param, item.ulmo, self.eps
                    )
                    if dparam > self.eps and grad_dual > self.eps:
                        l1hat = (dgrad / dparam) / grad_dual
                        self._append(stats, "l1", l1hat)
                        self._append(
                            stats, "eta_pred", self.action_scale / l1hat
                        )
                        eta = current_etas.get(item.group, float("nan"))
                        self._append(stats, "action_eff", eta * l1hat)

                input_sr = self.input_sr.get(id(item.param))
                if (
                    input_sr is not None
                    and grad.ndim == 2
                    and is_spectral_ulmo(item.ulmo)
                ):
                    ratio = nuclear_rank(current_grad, item.ulmo, self.eps) / max(
                        input_sr, self.eps
                    )
                    self._append(stats, "spec_ratio", ratio)

            self.prev[id(item.param)] = (current_grad.clone(), current_param.clone())
        self.active = False
        return self._format(step, grouped) if report else ""

    def _append(self, stats: dict[str, list[float]], name: str, value: float) -> None:
        if math.isfinite(value):
            stats.setdefault(name, []).append(value)

    def _format(self, step: int, grouped: dict[str, dict[str, list[float]]]) -> str:
        parts = []
        self.summary = {}
        for name, stats in grouped.items():
            fields = [f"eta={median(stats.get('eta', [])):.2e}"]
            if stats.get("l1"):
                l1 = median(stats["l1"])
                action_eff = median(stats["action_eff"])
                eta_pred = median(stats["eta_pred"])
                self.summary[name] = {
                    "l1": l1,
                    "action_eff": action_eff,
                    "eta_pred": eta_pred,
                }
                fields.append(f"L1={l1:.2e}")
                fields.append(f"act={action_eff:.2f}")
                fields.append(f"eta*={eta_pred:.2e}")
            fields.append(f"g*={median(stats.get('gdual', [])):.2e}")
            if stats.get("spec_ratio"):
                spec_ratio = median(stats["spec_ratio"])
                self.summary.setdefault(name, {})["spec_ratio"] = spec_ratio
                fields.append(f"Rspec={spec_ratio:.2f}")
            parts.append(f"{name}: " + ",".join(fields))
        if not parts:
            return ""
        return f"conv_stats step {step:5d} | " + "; ".join(parts)
