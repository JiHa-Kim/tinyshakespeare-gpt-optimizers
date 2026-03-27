import argparse
import copy
import csv
from pathlib import Path

from scion import scion_transfer_lr
from train_shakespeare import make_parser, train


def exp2_grid(exp2_min: float, exp2_max: float, step: float):
    if step <= 0:
        raise ValueError("step must be > 0")
    vals = []
    x = exp2_min
    while x <= exp2_max + 1e-12:
        vals.append(round(x, 10))
        x += step
    return vals


def dedupe_sorted(xs):
    out = []
    seen = set()
    for x in sorted(xs):
        k = round(x, 10)
        if k not in seen:
            seen.add(k)
            out.append(k)
    return out


def has_flag(rest, *names):
    return any(name in rest for name in names)


def apply_proxy_defaults(base, rest, args):
    if args.no_proxy:
        return
    if not has_flag(rest, "--n-layer"):
        base.n_layer = args.proxy_n_layer
    if not has_flag(rest, "--n-head"):
        base.n_head = args.proxy_n_head
    if not has_flag(rest, "--d-model"):
        base.d_model = args.proxy_d_model
    if not has_flag(rest, "--max-iters"):
        base.max_iters = args.proxy_max_iters
    if not has_flag(rest, "--eval-interval"):
        base.eval_interval = args.proxy_eval_interval
    if not has_flag(rest, "--eval-iters"):
        base.eval_iters = args.proxy_eval_iters


def token_budget(cfg):
    return cfg.max_iters * cfg.batch_size * cfg.grad_accum * cfg.block_size


def transfer_lrs(proxy_lr: float, proxy_cfg, target_cfg, alpha: float):
    mT = token_budget(target_cfg) / max(token_budget(proxy_cfg), 1)
    mL = target_cfg.n_layer / max(proxy_cfg.n_layer, 1)
    per_group = scion_transfer_lr(proxy_lr, mT=mT, mL=mL, alpha=alpha)
    return mT, mL, per_group


def run_sweep(
    base,
    exp2s,
    stage_name: str,
    keep_checkpoints: bool,
    prenorm: str,
    proxy_cfg,
    target_cfg,
    alpha: float,
):
    rows = []
    for i, exp2_lr in enumerate(exp2s):
        lr = 2.0**exp2_lr
        run = copy.deepcopy(base)
        run.lr = lr
        run.prenorm = prenorm
        run.seed = base.seed + i
        run.skip_sample = True
        run.no_save = not keep_checkpoints
        out_path = Path(base.out_path)
        run.out_path = str(
            out_path.with_name(
                f"{out_path.stem}_{prenorm}_{stage_name}_{i}_2p{exp2_lr:+.2f}{out_path.suffix}"
            )
        )
        print(
            f"=== {prenorm} | {stage_name} | 2**{exp2_lr:.2f} = {lr:.3e} ({i + 1}/{len(exp2s)}) ==="
        )
        metrics = train(run)
        mT, mL, per_group = transfer_lrs(lr, proxy_cfg, target_cfg, alpha)
        stable = (not metrics["diverged"]) and metrics["best_val"] == metrics[
            "best_val"
        ]
        row = {
            "prenorm": prenorm,
            "stage": stage_name,
            "exp2_lr": exp2_lr,
            "lr": lr,
            "stable": stable,
            "diverged": metrics["diverged"],
            "diverge_reason": metrics["diverge_reason"],
            "initial_val": metrics["initial_val"],
            "best_val": metrics["best_val"],
            "final_val": metrics["final_val"],
            "final_train": metrics["final_train"],
            "max_val": metrics["max_val"],
            "compile_seconds": metrics["compile_seconds"],
            "proxy_n_layer": proxy_cfg.n_layer,
            "proxy_n_head": proxy_cfg.n_head,
            "proxy_d_model": proxy_cfg.d_model,
            "proxy_max_iters": proxy_cfg.max_iters,
            "target_n_layer": target_cfg.n_layer,
            "target_n_head": target_cfg.n_head,
            "target_d_model": target_cfg.d_model,
            "target_max_iters": target_cfg.max_iters,
            "rho_embed": getattr(target_cfg, "rho_embed", float("nan")),
            "rho_hidden": getattr(target_cfg, "rho_hidden", float("nan")),
            "rho_out": getattr(target_cfg, "rho_out", float("nan")),
            "mT": mT,
            "mL": mL,
            "target_embed_lr": per_group["embed"],
            "target_hidden_lr": per_group["hidden"],
            "target_out_lr": per_group["out"],
            "out_path": run.out_path,
        }
        rows.append(row)
        status = "stable" if stable else f"diverged:{metrics['diverge_reason']}"
        print(
            f">>> {prenorm} | {stage_name} | 2**{exp2_lr:.2f} = {lr:.3e} "
            f"| {status} | best_val {metrics['best_val']:.4f} | target_hidden_lr {per_group['hidden']:.3e}"
        )
    return rows


def choose_refinement_center(rows):
    stable = [r for r in rows if r["stable"]]
    if stable:
        stable.sort(key=lambda r: (-r["exp2_lr"], r["best_val"]))
        return stable[0]["exp2_lr"]
    finite = [r for r in rows if r["best_val"] == r["best_val"]]
    finite.sort(key=lambda r: (r["best_val"], r["exp2_lr"]))
    return finite[0]["exp2_lr"]


def choose_recommendation(rows):
    stable = [r for r in rows if r["stable"]]
    if stable:
        stable.sort(key=lambda r: (-r["exp2_lr"], r["best_val"]))
        return stable[0], "stable-max-lr"
    finite = [r for r in rows if r["best_val"] == r["best_val"]]
    finite.sort(key=lambda r: (r["best_val"], r["exp2_lr"]))
    return (finite[0] if finite else rows[0]), "fallback-best-val"


def sweep_prenorm(base, target_cfg, args, coarse_exp2s, keep_checkpoints, prenorm):
    proxy_cfg = copy.deepcopy(base)
    coarse_rows = run_sweep(
        proxy_cfg,
        coarse_exp2s,
        "coarse",
        keep_checkpoints,
        prenorm,
        proxy_cfg,
        target_cfg,
        args.alpha_transfer,
    )
    center = choose_refinement_center(coarse_rows)
    fine_exp2s = exp2_grid(
        center - args.fine_radius, center + args.fine_radius, args.fine_step
    )
    fine_exp2s = [
        x
        for x in dedupe_sorted(fine_exp2s)
        if round(x, 10) not in {round(y, 10) for y in coarse_exp2s}
    ]
    fine_rows = (
        run_sweep(
            proxy_cfg,
            fine_exp2s,
            "fine",
            keep_checkpoints,
            prenorm,
            proxy_cfg,
            target_cfg,
            args.alpha_transfer,
        )
        if fine_exp2s
        else []
    )
    rows = coarse_rows + fine_rows
    rec, mode = choose_recommendation(rows)
    return rows, rec, center, mode


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--exp2-min", type=float, default=-14.0)
    p.add_argument("--exp2-max", type=float, default=-8.0)
    p.add_argument("--coarse-step", type=float, default=1.0)
    p.add_argument("--fine-step", type=float, default=0.25)
    p.add_argument("--fine-radius", type=float, default=0.0)
    p.add_argument("--csv-path", default="out/lr_sweep.csv")
    p.add_argument("--keep-checkpoints", action="store_true")
    p.add_argument("--no-proxy", action="store_true")
    p.add_argument("--proxy-n-layer", type=int, default=4)
    p.add_argument("--proxy-n-head", type=int, default=4)
    p.add_argument("--proxy-d-model", type=int, default=256)
    p.add_argument("--proxy-max-iters", type=int, default=600)
    p.add_argument("--proxy-eval-interval", type=int, default=50)
    p.add_argument("--proxy-eval-iters", type=int, default=20)
    p.add_argument("--alpha-transfer", type=float, default=0.5)
    p.add_argument("--prenorm", choices=["rmsnorm", "rmsball", "both"], default="both")
    args, rest = p.parse_known_args()

    target_cfg = make_parser().parse_args(rest)
    base = copy.deepcopy(target_cfg)
    apply_proxy_defaults(base, rest, args)

    coarse_exp2s = exp2_grid(args.exp2_min, args.exp2_max, args.coarse_step)
    all_rows = []
    recommendations = []
    prenorms = ["rmsnorm", "rmsball"] if args.prenorm == "both" else [args.prenorm]
    for prenorm in prenorms:
        rows, rec, center, mode = sweep_prenorm(
            base, target_cfg, args, coarse_exp2s, args.keep_checkpoints, prenorm
        )
        all_rows.extend(rows)
        recommendations.append((prenorm, rec, center, mode))

    csv_path = Path(args.csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        writer.writeheader()
        writer.writerows(all_rows)

    print("\n=== recommendations ===")
    for prenorm, rec, center, mode in recommendations:
        print(
            f"{prenorm:>7s} | center 2**{center:.2f} | proxy_lr 2**{rec['exp2_lr']:.2f} = {rec['lr']:.3e} "
            f"| {mode} | target_hidden_lr {rec['target_hidden_lr']:.3e} "
            f"| target_embed_lr {rec['target_embed_lr']:.3e} | target_out_lr {rec['target_out_lr']:.3e}"
        )
    print(f"csv_path {csv_path}")


if __name__ == "__main__":
    main()
