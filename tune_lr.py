import argparse
import copy
import csv
import math
from pathlib import Path

import torch

from gpt import BatchSource, CharDataset, GPT, GPTConfig, maybe_download_tiny_shakespeare
from scion import scion_transfer_lr
from train_shakespeare import (
    amp_ctx,
    build_optimizer,
    estimate_val_loss,
    init_gpt_scion_,
    make_parser,
    sync_now,
)


# -----------------------------
# generic utilities
# -----------------------------


def exp2_grid(exp2_min: float, exp2_max: float, step: float):
    if step <= 0:
        raise ValueError('step must be > 0')
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



def median(xs):
    xs = sorted(xs)
    if not xs:
        return float('nan')
    n = len(xs)
    if n % 2:
        return xs[n // 2]
    return 0.5 * (xs[n // 2 - 1] + xs[n // 2])



def has_flag(rest, *names):
    return any(name in rest for name in names)



def apply_proxy_defaults(base, rest, args):
    if args.no_proxy:
        return
    if not has_flag(rest, '--n-layer'):
        base.n_layer = args.proxy_n_layer
    if not has_flag(rest, '--n-head'):
        base.n_head = args.proxy_n_head
    if not has_flag(rest, '--d-model'):
        base.d_model = args.proxy_d_model
    if not has_flag(rest, '--max-iters'):
        base.max_iters = args.proxy_max_iters
    if not has_flag(rest, '--eval-interval'):
        base.eval_interval = args.proxy_eval_interval
    if not has_flag(rest, '--eval-iters'):
        base.eval_iters = args.proxy_eval_iters
    if not has_flag(rest, '--compile', '--no-compile'):
        base.compile = False



def parse_float_list(s: str):
    vals = []
    for part in s.split(','):
        part = part.strip()
        if not part:
            continue
        vals.append(float(part))
    if not vals:
        raise ValueError('expected at least one float')
    return vals



def token_budget(cfg):
    return cfg.max_iters * cfg.batch_size * cfg.grad_accum * cfg.block_size



def transfer_lrs(proxy_lr: float, proxy_cfg, target_cfg, alpha: float):
    mT = token_budget(target_cfg) / max(token_budget(proxy_cfg), 1)
    mL = target_cfg.n_layer / max(proxy_cfg.n_layer, 1)
    per_group = scion_transfer_lr(proxy_lr, mT=mT, mL=mL, alpha=alpha)
    return mT, mL, per_group



def score_with_tolerance(best_score: float, cand_score: float, rel_tol: float) -> bool:
    return cand_score <= best_score * (1.0 + rel_tol)



def clone_state_dict(state_dict, device: torch.device | None = None):
    out = {}
    for k, v in state_dict.items():
        if torch.is_tensor(v):
            dst = v.device if device is None else device
            out[k] = v.detach().to(dst).clone()
        elif isinstance(v, dict):
            out[k] = clone_state_dict(v, device=device)
        elif isinstance(v, list):
            out[k] = [
                clone_state_dict(x, device=device)
                if isinstance(x, dict)
                else (x.detach().to(x.device if device is None else device).clone() if torch.is_tensor(x) else copy.deepcopy(x))
                for x in v
            ]
        else:
            out[k] = copy.deepcopy(v)
    return out


def optimizer_to_device(opt: torch.optim.Optimizer, device: torch.device):
    for state in opt.state.values():
        for k, v in state.items():
            if torch.is_tensor(v):
                state[k] = v.to(device)



def set_seed(seed: int):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)





_SOURCE_CACHE = {}


def get_batch_source(dataset, args, device):
    key = (id(dataset.train), id(dataset.val), args.block_size, args.batch_size, str(device))
    source = _SOURCE_CACHE.get(key)
    if source is None:
        source = BatchSource(dataset.train, dataset.val, args.block_size, args.batch_size, device)
        _SOURCE_CACHE[key] = source
    return source


def prepare_runtime(base_args):
    torch.set_float32_matmul_precision('high')
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
    device = torch.device(base_args.device or ('cuda' if torch.cuda.is_available() else 'cpu'))
    amp_dtype = torch.bfloat16 if device.type == 'cuda' and torch.cuda.is_bf16_supported() else None
    data_path = Path(base_args.data_path)
    maybe_download_tiny_shakespeare(data_path)
    dataset = CharDataset(data_path)
    return device, amp_dtype, dataset



def build_run_objects(args, dataset, device):
    raw_model = GPT(
        GPTConfig(
            vocab_size=len(dataset.chars),
            block_size=args.block_size,
            n_layer=args.n_layer,
            n_head=args.n_head,
            d_model=args.d_model,
            rope_base=args.rope_base,
            prenorm=args.prenorm,
        )
    ).to(device)
    init_gpt_scion_(raw_model, args)
    source = get_batch_source(dataset, args, device)
    opt = build_optimizer(raw_model, args, device)
    opt._cpu_snapshots = getattr(args, 'cpu_snapshots', False)
    return raw_model, raw_model, opt, source


# -----------------------------
# low-level training loops
# -----------------------------


def run_loop(
    *,
    model,
    raw_model,
    opt,
    source,
    amp_dtype,
    device,
    steps: int,
    peak_lr: float,
    min_lr: float,
    eval_interval: int,
    eval_iters: int,
    grad_accum: int,
    grad_clip: float,
    diverge_mult: float,
    mode: str,
    snapshot_steps=None,
):
    if mode not in {'constant', 'decay'}:
        raise ValueError(f'invalid mode: {mode}')
    snapshot_steps = set(snapshot_steps or [])

    def lr_fn(step: int) -> float:
        if mode == 'constant' or steps <= 0:
            return peak_lr
        if steps == 1:
            return min_lr
        progress = step / (steps - 1)
        return peak_lr + (min_lr - peak_lr) * progress

    history = []
    snapshots = {}
    best_val = float('inf')
    max_val = float('-inf')
    last_losses = {'train': float('nan'), 'val': float('nan')}
    initial_val = None
    diverged = False
    diverge_reason = ''
    train_start = sync_now(device)
    effective_tokens = grad_accum * source.batch_size * source.block_size
    total_opt_steps = 0

    for step in range(steps):
        lr = lr_fn(step)
        for group in opt.param_groups:
            group['lr'] = lr

        if step % eval_interval == 0 or step == steps - 1:
            train_loss = last_losses['train']
            val_loss = estimate_val_loss(model, source, eval_iters, amp_dtype)
            last_losses['val'] = val_loss
            if not math.isfinite(val_loss):
                diverged, diverge_reason = True, 'nonfinite_eval_loss'
            else:
                if initial_val is None:
                    initial_val = val_loss
                best_val = min(best_val, val_loss)
                max_val = max(max_val, val_loss)
                if step > 0 and val_loss > initial_val * diverge_mult:
                    diverged = True
                    diverge_reason = f'val_loss_exceeded_{diverge_mult:.2f}x_initial'
            history.append(
                {
                    'step': step,
                    'lr': lr,
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    'best_val_so_far': best_val,
                }
            )
            elapsed = max(sync_now(device) - train_start, 1e-9)
            print(
                f'step {step:5d} | mode {mode:8s} | lr {lr:.3e} | train {train_loss:.4f} | val {val_loss:.4f} | '
                f'best_val {best_val:.4f} | train_seconds {elapsed:.3f} | tok/s {(total_opt_steps * effective_tokens) / elapsed:.0f}'
            )
            if diverged:
                print(f'diverged {diverge_reason}')
                break

        opt.zero_grad(set_to_none=True)
        train_loss = 0.0
        for _ in range(grad_accum):
            with amp_ctx(amp_dtype):
                _, loss = model(*source.get('train'))
                loss = loss / grad_accum
            loss_value = float(loss.detach())
            if not math.isfinite(loss_value):
                diverged, diverge_reason = True, 'nonfinite_train_loss'
                break
            train_loss += loss_value
            loss.backward()
        if diverged:
            print(f'diverged {diverge_reason}')
            break
        last_losses['train'] = train_loss

        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(raw_model.parameters(), grad_clip)
        opt.step()
        total_opt_steps += 1

        completed_steps = step + 1
        if completed_steps in snapshot_steps:
            snapshot_device = torch.device('cpu') if getattr(opt, '_cpu_snapshots', False) else None
            snapshots[completed_steps] = {
                'model': clone_state_dict(raw_model.state_dict(), device=snapshot_device),
                'opt': clone_state_dict(opt.state_dict(), device=snapshot_device),
                'completed_steps': completed_steps,
            }

    tail_window = history[-2:] if len(history) >= 2 else history
    tail_avg = float('nan')
    if tail_window:
        vals = [x['val_loss'] for x in tail_window if math.isfinite(x['val_loss'])]
        if vals:
            tail_avg = sum(vals) / len(vals)

    return {
        'best_val': best_val,
        'final_val': last_losses['val'],
        'final_train': last_losses['train'],
        'initial_val': float('nan') if initial_val is None else initial_val,
        'max_val': max_val,
        'tail_avg_val': tail_avg,
        'diverged': diverged,
        'diverge_reason': diverge_reason,
        'history': history,
        'snapshots': snapshots,
        'completed_steps': total_opt_steps,
    }


# -----------------------------
# screen stage
# -----------------------------


def make_screen_args(base_args, prenorm: str, steps: int, eval_interval: int, eval_iters: int, compile_flag: bool):
    args = copy.deepcopy(base_args)
    args.prenorm = prenorm
    args.max_iters = steps
    args.eval_interval = eval_interval
    args.eval_iters = eval_iters
    args.compile = compile_flag
    args.skip_sample = True
    args.no_save = True
    args.warmup_frac = 0.0
    args.warmup_iters = 0
    args.decay_frac = 0.0
    args.min_lr = args.lr if hasattr(args, 'lr') else 0.0
    return args



def screen_candidate(exp2_lr, args, dataset, device, amp_dtype, seed):
    lr = 2.0**exp2_lr
    set_seed(seed)
    run_args = copy.deepcopy(args)
    run_args.lr = lr
    raw_model, model, opt, source = build_run_objects(run_args, dataset, device)
    print(f'=== screen | {run_args.prenorm} | seed {seed} | 2**{exp2_lr:.2f} = {lr:.3e} ===')
    metrics = run_loop(
        model=model,
        raw_model=raw_model,
        opt=opt,
        source=source,
        amp_dtype=amp_dtype,
        device=device,
        steps=run_args.max_iters,
        peak_lr=lr,
        min_lr=lr,
        eval_interval=run_args.eval_interval,
        eval_iters=run_args.eval_iters,
        grad_accum=run_args.grad_accum,
        grad_clip=run_args.grad_clip,
        diverge_mult=run_args.diverge_mult,
        mode='constant',
    )
    stable = (not metrics['diverged']) and math.isfinite(metrics['final_val'])
    row = {
        'stage': 'screen',
        'prenorm': run_args.prenorm,
        'seed': seed,
        'exp2_lr': exp2_lr,
        'lr': lr,
        'stable': stable,
        'diverged': metrics['diverged'],
        'diverge_reason': metrics['diverge_reason'],
        'initial_val': metrics['initial_val'],
        'best_val': metrics['best_val'],
        'final_val': metrics['final_val'],
        'tail_avg_val': metrics['tail_avg_val'],
        'max_val': metrics['max_val'],
    }
    return row



def choose_screen_center(rows):
    stable = [r for r in rows if r['stable'] and math.isfinite(r['final_val'])]
    if stable:
        stable.sort(key=lambda r: (r['final_val'], r['exp2_lr']))
        return stable[0]['exp2_lr']
    finite = [r for r in rows if math.isfinite(r['final_val'])]
    finite.sort(key=lambda r: (r['final_val'], r['exp2_lr']))
    return finite[0]['exp2_lr'] if finite else rows[0]['exp2_lr']



def auto_expand_screen(screen_args, dataset, device, amp_dtype, screen_seed, args):
    cache = {}
    exp2_min = args.exp2_min
    exp2_max = args.exp2_max

    for _ in range(args.screen_max_expansions + 1):
        for exp2_lr in exp2_grid(exp2_min, exp2_max, args.coarse_step):
            if exp2_lr not in cache:
                cache[exp2_lr] = screen_candidate(exp2_lr, screen_args, dataset, device, amp_dtype, screen_seed)

        rows = [cache[k] for k in sorted(cache)]
        stable = [r for r in rows if r['stable']]
        if stable:
            best = min(stable, key=lambda r: r['final_val'])
            top = max(stable, key=lambda r: r['exp2_lr'])
            low = min(stable, key=lambda r: r['exp2_lr'])
            expanded = False
            if round(top['exp2_lr'], 10) == round(exp2_max, 10) and score_with_tolerance(best['final_val'], top['final_val'], args.edge_tol):
                exp2_min, exp2_max = exp2_min, exp2_max + args.coarse_step * args.expand_points
                expanded = True
            elif round(low['exp2_lr'], 10) == round(exp2_min, 10) and score_with_tolerance(best['final_val'], low['final_val'], args.edge_tol):
                exp2_min, exp2_max = exp2_min - args.coarse_step * args.expand_points, exp2_max
                expanded = True
            if not expanded:
                return rows
        else:
            exp2_min -= args.coarse_step * args.expand_points
            exp2_max += args.coarse_step * args.expand_points
    return [cache[k] for k in sorted(cache)]



def shortlist_from_screen(rows, shortlist_k: int, keep_highest_k: int):
    stable = [r for r in rows if r['stable'] and math.isfinite(r['final_val'])]
    if not stable:
        stable = [r for r in rows if math.isfinite(r['final_val'])]
    by_quality = sorted(stable, key=lambda r: (r['final_val'], r['exp2_lr']))[:shortlist_k]
    by_aggressive = sorted(stable, key=lambda r: (-r['exp2_lr'], r['final_val']))[:keep_highest_k]
    chosen = {round(r['exp2_lr'], 10): r for r in by_quality + by_aggressive}
    return [chosen[k] for k in sorted(chosen)]


# -----------------------------
# branch stage
# -----------------------------


def make_branch_args(base_args, prenorm: str, steps: int, eval_interval: int, eval_iters: int, compile_flag: bool):
    args = copy.deepcopy(base_args)
    args.prenorm = prenorm
    args.max_iters = steps
    args.eval_interval = eval_interval
    args.eval_iters = eval_iters
    args.compile = compile_flag
    args.skip_sample = True
    args.no_save = True
    args.warmup_frac = 0.0
    args.warmup_iters = 0
    return args



def run_branch_family(exp2_lr, branch_args, dataset, device, amp_dtype, seed, decay_fracs):
    lr = 2.0**exp2_lr
    total_steps = branch_args.max_iters
    decay_steps_map = {df: max(1, round(df * total_steps)) for df in decay_fracs}
    branch_start_steps = {df: total_steps - ds for df, ds in decay_steps_map.items()}
    snapshot_steps = set(branch_start_steps.values())

    set_seed(seed)
    trunk_args = copy.deepcopy(branch_args)
    trunk_args.lr = lr
    raw_model, model, opt, source = build_run_objects(trunk_args, dataset, device)
    print(f'=== branch trunk | {trunk_args.prenorm} | seed {seed} | 2**{exp2_lr:.2f} = {lr:.3e} ===')
    trunk_metrics = run_loop(
        model=model,
        raw_model=raw_model,
        opt=opt,
        source=source,
        amp_dtype=amp_dtype,
        device=device,
        steps=total_steps,
        peak_lr=lr,
        min_lr=lr,
        eval_interval=trunk_args.eval_interval,
        eval_iters=trunk_args.eval_iters,
        grad_accum=trunk_args.grad_accum,
        grad_clip=trunk_args.grad_clip,
        diverge_mult=trunk_args.diverge_mult,
        mode='constant',
        snapshot_steps=snapshot_steps,
    )

    rows = []
    if trunk_metrics['diverged']:
        for df in decay_fracs:
            rows.append(
                {
                    'stage': 'branch',
                    'prenorm': trunk_args.prenorm,
                    'seed': seed,
                    'exp2_lr': exp2_lr,
                    'lr': lr,
                    'decay_frac': df,
                    'branch_start_step': branch_start_steps[df],
                    'tail_steps': decay_steps_map[df],
                    'stable': False,
                    'diverged': True,
                    'diverge_reason': f"trunk_{trunk_metrics['diverge_reason']}",
                    'trunk_final_val': trunk_metrics['final_val'],
                    'trunk_best_val': trunk_metrics['best_val'],
                    'tail_final_val': float('nan'),
                    'tail_best_val': float('nan'),
                    'tail_avg_val': float('nan'),
                }
            )
        return rows

    tail_args = copy.deepcopy(branch_args)
    tail_args.lr = lr
    raw_tail, tail_model, tail_opt, tail_source = build_run_objects(tail_args, dataset, device)

    for df in decay_fracs:
        snap = trunk_metrics['snapshots'][branch_start_steps[df]]
        raw_tail.load_state_dict(snap['model'])
        tail_opt.load_state_dict(snap['opt'])
        if getattr(tail_opt, '_cpu_snapshots', False):
            optimizer_to_device(tail_opt, device)
        print(
            f'=== branch tail | {tail_args.prenorm} | seed {seed} | 2**{exp2_lr:.2f} = {lr:.3e} | '
            f'decay_frac {df:.3f} | start_step {branch_start_steps[df]} ==='
        )
        tail_metrics = run_loop(
            model=tail_model,
            raw_model=raw_tail,
            opt=tail_opt,
            source=tail_source,
            amp_dtype=amp_dtype,
            device=device,
            steps=decay_steps_map[df],
            peak_lr=lr,
            min_lr=tail_args.min_lr,
            eval_interval=max(1, min(tail_args.eval_interval, decay_steps_map[df] // 3 if decay_steps_map[df] >= 3 else 1)),
            eval_iters=tail_args.eval_iters,
            grad_accum=tail_args.grad_accum,
            grad_clip=tail_args.grad_clip,
            diverge_mult=tail_args.diverge_mult,
            mode='decay',
        )
        stable = (not tail_metrics['diverged']) and math.isfinite(tail_metrics['final_val'])
        rows.append(
            {
                'stage': 'branch',
                'prenorm': tail_args.prenorm,
                'seed': seed,
                'exp2_lr': exp2_lr,
                'lr': lr,
                'decay_frac': df,
                'branch_start_step': branch_start_steps[df],
                'tail_steps': decay_steps_map[df],
                'stable': stable,
                'diverged': tail_metrics['diverged'],
                'diverge_reason': tail_metrics['diverge_reason'],
                'trunk_final_val': trunk_metrics['final_val'],
                'trunk_best_val': trunk_metrics['best_val'],
                'tail_final_val': tail_metrics['final_val'],
                'tail_best_val': tail_metrics['best_val'],
                'tail_avg_val': tail_metrics['tail_avg_val'],
            }
        )
    return rows



def summarize_branch_rows(rows):
    pair_rows = {}
    lr_rows = {}

    by_pair = {}
    for row in rows:
        key = (round(row['exp2_lr'], 10), round(row['decay_frac'], 10))
        by_pair.setdefault(key, []).append(row)

    for key, group in by_pair.items():
        finals = [r['tail_final_val'] for r in group if math.isfinite(r['tail_final_val'])]
        avgs = [r['tail_avg_val'] for r in group if math.isfinite(r['tail_avg_val'])]
        stable_rate = sum(1 for r in group if r['stable']) / len(group)
        rep = dict(group[0])
        rep.update(
            {
                'trial_count': len(group),
                'stable_rate': stable_rate,
                'median_tail_final_val': median(finals),
                'median_tail_avg_val': median(avgs),
                'score': median(finals),
            }
        )
        pair_rows[key] = rep

    by_lr = {}
    for pair in pair_rows.values():
        by_lr.setdefault(round(pair['exp2_lr'], 10), []).append(pair)

    for lr_key, group in by_lr.items():
        scores = [g['score'] for g in group if math.isfinite(g['score'])]
        robust = median(scores)
        best_pair = min(group, key=lambda g: (g['score'], g['decay_frac']))
        rep = dict(best_pair)
        rep.update(
            {
                'pair_count': len(group),
                'potential_score': best_pair['score'],
                'robust_score': robust,
                'best_decay_frac': best_pair['decay_frac'],
                'best_pair_score': best_pair['score'],
                'min_stable_rate_across_pairs': min(g['stable_rate'] for g in group),
            }
        )
        lr_rows[lr_key] = rep

    return list(pair_rows.values()), list(lr_rows.values())



def choose_recommendations(lr_summaries, pair_summaries, promote_tol: float, robust_tol: float):
    finite_pairs = [r for r in pair_summaries if math.isfinite(r['score'])]
    finite_lrs = [r for r in lr_summaries if math.isfinite(r['potential_score'])]
    best_pair = min(finite_pairs, key=lambda r: (r['score'], r['lr'], r['decay_frac']))
    best_lr_score = min(r['potential_score'] for r in finite_lrs)
    promote_frontier = [r for r in finite_lrs if score_with_tolerance(best_lr_score, r['potential_score'], promote_tol)]
    promoted_lr = max(promote_frontier, key=lambda r: (r['lr'], -r['robust_score']))
    best_robust = min(r['robust_score'] for r in finite_lrs)
    robust_frontier = [r for r in finite_lrs if score_with_tolerance(best_robust, r['robust_score'], robust_tol)]
    robust_lr = max(robust_frontier, key=lambda r: (r['lr'], -r['potential_score']))
    return best_pair, promoted_lr, robust_lr



def write_csv(path: Path, rows):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


# -----------------------------
# main orchestration
# -----------------------------


def run_for_prenorm(prenorm, target_cfg, args, rest):
    device, amp_dtype, dataset = prepare_runtime(target_cfg)

    proxy_cfg = copy.deepcopy(target_cfg)
    apply_proxy_defaults(proxy_cfg, rest, args)
    proxy_cfg.prenorm = prenorm
    proxy_cfg.compile = False if not args.compile_tuning else proxy_cfg.compile

    screen_args = make_screen_args(
        proxy_cfg,
        prenorm,
        steps=args.screen_steps,
        eval_interval=args.screen_eval_interval,
        eval_iters=args.screen_eval_iters,
        compile_flag=args.compile_tuning,
    )
    screen_seed = proxy_cfg.seed
    coarse_rows = auto_expand_screen(screen_args, dataset, device, amp_dtype, screen_seed, args)
    center = choose_screen_center(coarse_rows)
    fine_exp2s = [
        x for x in exp2_grid(center - args.fine_radius, center + args.fine_radius, args.fine_step)
        if round(x, 10) not in {round(r['exp2_lr'], 10) for r in coarse_rows}
    ]
    fine_rows = [screen_candidate(x, screen_args, dataset, device, amp_dtype, screen_seed) for x in fine_exp2s]
    screen_rows = sorted(coarse_rows + fine_rows, key=lambda r: r['exp2_lr'])

    shortlist = shortlist_from_screen(screen_rows, args.shortlist_k, args.keep_highest_k)
    shortlist_exp2s = [r['exp2_lr'] for r in shortlist]

    branch_args = make_branch_args(
        proxy_cfg,
        prenorm,
        steps=args.branch_steps,
        eval_interval=args.branch_eval_interval,
        eval_iters=args.branch_eval_iters,
        compile_flag=args.compile_tuning,
    )
    decay_fracs = dedupe_sorted(parse_float_list(args.decay_fracs))

    branch_raw = []
    for exp2_lr in shortlist_exp2s:
        branch_raw.extend(run_branch_family(exp2_lr, branch_args, dataset, device, amp_dtype, branch_args.seed, decay_fracs))

    pair_summaries, lr_summaries = summarize_branch_rows(branch_raw)
    best_pair, promoted_lr, robust_lr = choose_recommendations(lr_summaries, pair_summaries, args.promote_tol, args.robust_tol)

    candidate_pool = sorted(lr_summaries, key=lambda r: (r['potential_score'], -r['lr']))
    confirm_candidates = [round(best_pair['exp2_lr'], 10), round(promoted_lr['exp2_lr'], 10), round(robust_lr['exp2_lr'], 10)]
    for row in candidate_pool[: args.confirm_top_k]:
        confirm_candidates.append(round(row['exp2_lr'], 10))
    confirm_candidates = dedupe_sorted(confirm_candidates)

    confirm_raw = []
    if args.confirm_num_seeds > 0:
        for exp2_lr in confirm_candidates:
            for seed_idx in range(args.confirm_num_seeds):
                seed = branch_args.seed + seed_idx * args.seed_stride
                confirm_raw.extend(run_branch_family(exp2_lr, branch_args, dataset, device, amp_dtype, seed, decay_fracs))
        final_pair_summaries, final_lr_summaries = summarize_branch_rows(confirm_raw)
        final_best_pair, final_promoted_lr, final_robust_lr = choose_recommendations(
            final_lr_summaries, final_pair_summaries, args.promote_tol, args.robust_tol
        )
    else:
        final_pair_summaries, final_lr_summaries = pair_summaries, lr_summaries
        final_best_pair, final_promoted_lr, final_robust_lr = best_pair, promoted_lr, robust_lr

    summary_rows = [
        {
            'prenorm': prenorm,
            'recommendation_type': 'best_pair',
            'exp2_lr': final_best_pair['exp2_lr'],
            'lr': final_best_pair['lr'],
            'decay_frac': final_best_pair['decay_frac'],
            'score': final_best_pair['score'],
            'score_kind': 'median_tail_final_val',
        },
        {
            'prenorm': prenorm,
            'recommendation_type': 'promoted_lr',
            'exp2_lr': final_promoted_lr['exp2_lr'],
            'lr': final_promoted_lr['lr'],
            'decay_frac': final_promoted_lr['best_decay_frac'],
            'score': final_promoted_lr['potential_score'],
            'score_kind': 'best_pair_score_per_lr',
        },
        {
            'prenorm': prenorm,
            'recommendation_type': 'robust_lr',
            'exp2_lr': final_robust_lr['exp2_lr'],
            'lr': final_robust_lr['lr'],
            'decay_frac': final_robust_lr['best_decay_frac'],
            'score': final_robust_lr['robust_score'],
            'score_kind': 'median_pair_score_per_lr',
        },
    ]

    for row in summary_rows:
        mT, mL, per_group = transfer_lrs(row['lr'], proxy_cfg, target_cfg, args.alpha_transfer)
        row['mT'] = mT
        row['mL'] = mL
        row['target_embed_lr'] = per_group['embed']
        row['target_hidden_lr'] = per_group['hidden']
        row['target_out_lr'] = per_group['out']

    return {
        'screen_rows': screen_rows,
        'branch_raw': branch_raw,
        'pair_summaries': pair_summaries,
        'lr_summaries': lr_summaries,
        'confirm_raw': confirm_raw,
        'final_pair_summaries': final_pair_summaries,
        'final_lr_summaries': final_lr_summaries,
        'summary_rows': summary_rows,
    }



def main():
    p = argparse.ArgumentParser()
    p.add_argument('--exp2-min', type=float, default=-14.0)
    p.add_argument('--exp2-max', type=float, default=-8.0)
    p.add_argument('--coarse-step', type=float, default=1.0)
    p.add_argument('--fine-step', type=float, default=0.25)
    p.add_argument('--fine-radius', type=float, default=1.0)
    p.add_argument('--expand-points', type=int, default=2)
    p.add_argument('--screen-max-expansions', type=int, default=3)
    p.add_argument('--edge-tol', type=float, default=0.03)

    p.add_argument('--screen-steps', type=int, default=200)
    p.add_argument('--screen-eval-interval', type=int, default=50)
    p.add_argument('--screen-eval-iters', type=int, default=10)
    p.add_argument('--shortlist-k', type=int, default=4)
    p.add_argument('--keep-highest-k', type=int, default=1)

    p.add_argument('--branch-steps', type=int, default=600)
    p.add_argument('--branch-eval-interval', type=int, default=50)
    p.add_argument('--branch-eval-iters', type=int, default=20)
    p.add_argument('--decay-fracs', default='0.10,0.20,0.285,0.40')

    p.add_argument('--confirm-top-k', type=int, default=3)
    p.add_argument('--confirm-num-seeds', type=int, default=3)
    p.add_argument('--seed-stride', type=int, default=1000)

    p.add_argument('--promote-tol', type=float, default=0.02)
    p.add_argument('--robust-tol', type=float, default=0.02)

    p.add_argument('--csv-prefix', default='out/wsd_tune')
    p.add_argument('--no-proxy', action='store_true')
    p.add_argument('--proxy-n-layer', type=int, default=4)
    p.add_argument('--proxy-n-head', type=int, default=4)
    p.add_argument('--proxy-d-model', type=int, default=256)
    p.add_argument('--proxy-max-iters', type=int, default=600)
    p.add_argument('--proxy-eval-interval', type=int, default=50)
    p.add_argument('--proxy-eval-iters', type=int, default=20)
    p.add_argument('--alpha-transfer', type=float, default=0.5)
    p.add_argument('--prenorm', choices=['rmsnorm', 'rmsball', 'both'], default='both')
    p.add_argument('--compile-tuning', action='store_true')
    p.add_argument('--cpu-snapshots', action='store_true', help='store branch snapshots on CPU instead of the active device')
    args, rest = p.parse_known_args()

    target_cfg = make_parser().parse_args(rest)
    if not has_flag(rest, '--optimizer'):
        target_cfg.optimizer = 'scion'
    if not has_flag(rest, '--warmup-frac', '--warmup-iters'):
        target_cfg.warmup_frac = 0.0
        target_cfg.warmup_iters = 0
    if not has_flag(rest, '--min-lr'):
        target_cfg.min_lr = 0.0
    if not has_flag(rest, '--grad-clip'):
        target_cfg.grad_clip = 0.0
    if not has_flag(rest, '--phi'):
        target_cfg.phi = 0.0

    prenorms = ['rmsnorm', 'rmsball'] if args.prenorm == 'both' else [args.prenorm]
    all_screen_rows = []
    all_branch_raw = []
    all_pair_summaries = []
    all_lr_summaries = []
    all_confirm_raw = []
    all_final_pair_summaries = []
    all_final_lr_summaries = []
    all_summary_rows = []

    for prenorm in prenorms:
        print(f'\n===== tuning prenorm={prenorm} =====')
        out = run_for_prenorm(prenorm, target_cfg, args, rest)
        all_screen_rows.extend(out['screen_rows'])
        all_branch_raw.extend(out['branch_raw'])
        all_pair_summaries.extend(out['pair_summaries'])
        all_lr_summaries.extend(out['lr_summaries'])
        all_confirm_raw.extend(out['confirm_raw'])
        all_final_pair_summaries.extend(out['final_pair_summaries'])
        all_final_lr_summaries.extend(out['final_lr_summaries'])
        all_summary_rows.extend(out['summary_rows'])

    prefix = Path(args.csv_prefix)
    write_csv(prefix.with_name(prefix.name + '_screen.csv'), all_screen_rows)
    write_csv(prefix.with_name(prefix.name + '_branch_raw.csv'), all_branch_raw)
    write_csv(prefix.with_name(prefix.name + '_pair_summary.csv'), all_pair_summaries)
    write_csv(prefix.with_name(prefix.name + '_lr_summary.csv'), all_lr_summaries)
    write_csv(prefix.with_name(prefix.name + '_confirm_raw.csv'), all_confirm_raw)
    write_csv(prefix.with_name(prefix.name + '_final_pair_summary.csv'), all_final_pair_summaries)
    write_csv(prefix.with_name(prefix.name + '_final_lr_summary.csv'), all_final_lr_summaries)
    write_csv(prefix.with_name(prefix.name + '_recommendations.csv'), all_summary_rows)

    print('\n=== final recommendations ===')
    for row in all_summary_rows:
        print(
            f"{row['prenorm']:>7s} | {row['recommendation_type']:>11s} | 2**{row['exp2_lr']:.2f} = {row['lr']:.3e} | "
            f"decay_frac {row['decay_frac']:.3f} | score {row['score']:.4f} | "
            f"target_hidden_lr {row['target_hidden_lr']:.3e}"
        )


if __name__ == '__main__':
    main()
