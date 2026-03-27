import argparse
import math
import time
from contextlib import nullcontext
from dataclasses import asdict
from pathlib import Path

import torch

from gpt import BatchSource, CharDataset, GPT, GPTConfig, maybe_download_tiny_shakespeare
from scion import (
    ColNormLMO,
    RowNormLMO,
    Scion,
    ScionC,
    SpectralLMO,
    init_colnorm_,
    init_rownorm_,
    init_spectral_,
)


def sync_now(device: torch.device) -> float:
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    return time.perf_counter()



def amp_ctx(amp_dtype: torch.dtype | None):
    return torch.autocast(device_type='cuda', dtype=amp_dtype) if amp_dtype is not None else nullcontext()



def resolve_schedule(max_steps: int, warmup_steps: int, decay_steps: int) -> tuple[int, int, int]:
    if max_steps <= 0:
        raise ValueError(f'invalid max_steps: {max_steps}')
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
    warmup_steps, stable_steps, decay_steps = resolve_schedule(max_steps, warmup_steps, decay_steps)

    if warmup_steps > 0 and step < warmup_steps:
        return lr * (step + 1) / warmup_steps

    decay_start = warmup_steps + stable_steps
    if decay_steps == 0 or step < decay_start:
        return lr
    if decay_steps == 1:
        return min_lr

    progress = (step - decay_start) / (decay_steps - 1)
    progress = min(max(progress, 0.0), 1.0)
    return lr + (min_lr - lr) * progress


@torch.inference_mode()
def estimate_loss(
    model: GPT,
    source: BatchSource,
    eval_iters: int,
    amp_dtype: torch.dtype | None,
    splits=('train', 'val'),
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
def estimate_val_loss(model: GPT, source: BatchSource, eval_iters: int, amp_dtype: torch.dtype | None) -> float:
    return estimate_loss(model, source, eval_iters, amp_dtype, splits=('val',))['val']


@torch.no_grad()
def init_gpt_scion_(model: GPT, args):
    init_colnorm_(model.tok_emb.weight, radius=args.rho_embed, transpose=True)
    for block in model.blocks:
        init_spectral_(block.attn.qkv.weight, radius=args.rho_hidden)
        init_spectral_(block.attn.proj.weight, radius=args.rho_hidden)
        init_spectral_(block.mlp.gate.weight, radius=args.rho_hidden)
        init_spectral_(block.mlp.up.weight, radius=args.rho_hidden)
        init_spectral_(block.mlp.down.weight, radius=args.rho_hidden)
    init_rownorm_(model.lm_head.weight, radius=args.rho_out)


@torch.no_grad()
def build_optimizer(model: GPT, args, device: torch.device):
    work_dtype = torch.bfloat16 if device.type == 'cuda' else None
    if (
        args.optimizer == 'scionc'
        and args.eta is None
        and args.theta2_embed is None
        and args.theta2_hidden is None
        and args.theta2_out is None
    ):
        print(
            'warning: optimizer=scionc but no eta/theta2 values were provided; '
            'corrected decay is effectively disabled.'
        )
    skip = {id(model.tok_emb.weight), id(model.lm_head.weight)}
    hidden = [p for p in model.parameters() if p.requires_grad and id(p) not in skip]
    groups = []

    def add(params, dir_fn, theta2=None):
        if not params:
            return
        group = {'params': params, 'dir_fn': dir_fn}
        if args.optimizer == 'scionc' and theta2 is not None:
            group['theta2'] = theta2
        groups.append(group)

    add([model.tok_emb.weight], ColNormLMO(args.rho_embed, transpose=True), args.theta2_embed)
    add(
        hidden,
        SpectralLMO(args.rho_hidden, args.pe_steps, work_dtype=work_dtype),
        args.theta2_hidden,
    )
    add([model.lm_head.weight], RowNormLMO(args.rho_out), args.theta2_out)

    opt_cls = Scion if args.optimizer == 'scion' else ScionC
    return opt_cls(
        groups,
        lr=args.lr,
        beta2=args.beta2,
        phi=args.phi,
        eta=args.eta,
        cwd=args.cwd,
        nesterov=args.nesterov,
    )


def save_checkpoint(path: Path, model: GPT, dataset: CharDataset, args):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            'model': model.state_dict(),
            'model_cfg': asdict(model.cfg),
            'chars': dataset.chars,
            'args': vars(args),
        },
        path,
    )


def load_checkpoint(path: Path, device: torch.device):
    ckpt = torch.load(path, map_location=device)
    chars = ckpt['chars']
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}
    model = GPT(GPTConfig(**ckpt['model_cfg'])).to(device)
    model.load_state_dict(ckpt['model'])
    return model, stoi, itos



def maybe_compile(model: GPT, source: BatchSource, args, amp_dtype: torch.dtype | None, device: torch.device):
    if not (args.compile and hasattr(torch, 'compile')):
        return model, 0.0
    model = torch.compile(model)
    xb, yb = source.get('train')
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
    torch.set_float32_matmul_precision('high')
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    device = torch.device(args.device or ('cuda' if torch.cuda.is_available() else 'cpu'))
    amp_dtype = torch.bfloat16 if device.type == 'cuda' and torch.cuda.is_bf16_supported() else None

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
        )
    ).to(device)
    init_gpt_scion_(raw_model, args)

    source = BatchSource(dataset.train, dataset.val, args.block_size, args.batch_size, device)
    opt = build_optimizer(raw_model, args, device)
    model, compile_seconds = maybe_compile(raw_model, source, args, amp_dtype, device)
    if compile_seconds:
        print(f'compile_seconds {compile_seconds:.3f}')

    warmup_steps = args.warmup_iters if args.warmup_iters >= 0 else round(args.warmup_frac * args.max_iters)
    decay_steps = round(args.decay_frac * args.max_iters)
    warmup_steps, stable_steps, decay_steps = resolve_schedule(args.max_iters, warmup_steps, decay_steps)
    effective_tokens = args.batch_size * args.block_size * args.grad_accum

    print(
        'schedule '
        f'warmup_steps={warmup_steps} stable_steps={stable_steps} decay_steps={decay_steps} '
        f'lr={args.lr:.3e} min_lr={args.min_lr:.3e} optimizer={args.optimizer} prenorm={args.prenorm}'
    )

    total_opt_steps = 0
    best_val = float('inf')
    max_val = float('-inf')
    last_losses = {'train': float('nan'), 'val': float('nan')}
    initial_val = None
    diverged = False
    diverge_reason = ''
    train_start = sync_now(device)

    for step in range(args.max_iters):
        lr = lr_at_step(step, args.max_iters, args.lr, args.min_lr, warmup_steps, decay_steps)
        for group in opt.param_groups:
            group['lr'] = lr

        if step % args.eval_interval == 0 or step == args.max_iters - 1:
            train_loss = last_losses['train']
            val_loss = estimate_val_loss(model, source, args.eval_iters, amp_dtype)
            last_losses['val'] = val_loss

            if not math.isfinite(val_loss):
                diverged, diverge_reason = True, 'nonfinite_eval_loss'
            else:
                prev_best = best_val
                if initial_val is None:
                    initial_val = val_loss
                best_val = min(best_val, val_loss)
                max_val = max(max_val, val_loss)
                if step > 0 and val_loss > initial_val * args.diverge_mult:
                    diverged = True
                    diverge_reason = f'val_loss_exceeded_{args.diverge_mult:.2f}x_initial'
                if not args.no_save and (val_loss < prev_best or step == args.max_iters - 1):
                    save_checkpoint(Path(args.out_path), raw_model, dataset, args)

            elapsed = max(sync_now(device) - train_start, 1e-9)
            print(
                f'step {step:5d} | lr {lr:.3e} | train {train_loss:.4f} | val {val_loss:.4f} | '
                f'best_val {best_val:.4f} | train_seconds {elapsed:.3f} | tok/s {(total_opt_steps * effective_tokens) / elapsed:.0f}'
            )
            if diverged:
                print(f'diverged {diverge_reason}')
                break

        opt.zero_grad(set_to_none=True)
        train_loss = 0.0
        for _ in range(args.grad_accum):
            with amp_ctx(amp_dtype):
                _, loss = model(*source.get('train'))
                loss = loss / args.grad_accum
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

        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(raw_model.parameters(), args.grad_clip)
        opt.step()
        total_opt_steps += 1

    if not (args.skip_sample or diverged):
        y = raw_model.generate(
            torch.tensor([dataset.encode(args.prompt or '\n')], dtype=torch.long, device=device),
            max_new_tokens=args.sample_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
        )
        print('\n--- sample ---\n')
        print(dataset.decode(y[0].tolist()))

    return {
        'best_val': best_val,
        'final_train': last_losses['train'],
        'final_val': last_losses['val'],
        'compile_seconds': compile_seconds,
        'initial_val': float('nan') if initial_val is None else initial_val,
        'max_val': max_val,
        'diverged': diverged,
        'diverge_reason': diverge_reason,
        'warmup_steps': warmup_steps,
        'stable_steps': stable_steps,
        'decay_steps': decay_steps,
    }


@torch.inference_mode()
def sample(args):
    device = torch.device(args.device or ('cuda' if torch.cuda.is_available() else 'cpu'))
    model, stoi, itos = load_checkpoint(Path(args.out_path), device)
    prompt = args.prompt or '\n'
    bad = [c for c in prompt if c not in stoi]
    if bad:
        raise ValueError(f'prompt contains unseen chars: {bad}')
    y = model.generate(
        torch.tensor([[stoi[c] for c in prompt]], dtype=torch.long, device=device),
        max_new_tokens=args.sample_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
    )
    print(''.join(itos[int(i)] for i in y[0].tolist()))



def make_parser():
    p = argparse.ArgumentParser()
    p.add_argument('--mode', choices=['train', 'sample'], default='train')
    p.add_argument('--data-path', default='data/tiny_shakespeare.txt')
    p.add_argument('--out-path', default='out/scion_shakespeare.pt')
    p.add_argument('--device', default='')
    p.add_argument('--seed', type=int, default=1337)
    p.add_argument('--compile', action=argparse.BooleanOptionalAction, default=True)

    p.add_argument('--block-size', type=int, default=256)
    p.add_argument('--batch-size', type=int, default=64, help='microbatch size')
    p.add_argument('--grad-accum', type=int, default=1)
    p.add_argument('--n-layer', type=int, default=6)
    p.add_argument('--n-head', type=int, default=6)
    p.add_argument('--d-model', type=int, default=384)
    p.add_argument('--rope-base', type=float, default=10000.0)
    p.add_argument('--prenorm', choices=['rmsnorm', 'rmsball'], default='rmsnorm')

    p.add_argument('--max-iters', type=int, default=3000)
    p.add_argument('--eval-interval', type=int, default=200)
    p.add_argument('--eval-iters', type=int, default=50)
    p.add_argument('--grad-clip', type=float, default=0.0)
    p.add_argument('--diverge-mult', type=float, default=2.0)

    p.add_argument('--warmup-iters', type=int, default=-1, help='if >=0, overrides warmup-frac')
    p.add_argument('--warmup-frac', type=float, default=0.0)
    p.add_argument('--decay-frac', type=float, default=0.285)

    p.add_argument('--optimizer', choices=['scion', 'scionc'], default='scion')
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--min-lr', type=float, default=0.0)
    p.add_argument('--beta2', type=float, default=0.95)
    p.add_argument('--phi', type=float, default=0.0)
    p.add_argument('--eta', type=float, default=None)
    p.add_argument('--cwd', action='store_true')
    p.add_argument('--nesterov', action=argparse.BooleanOptionalAction, default=True)
    p.add_argument('--theta2-embed', type=float, default=None)
    p.add_argument('--theta2-hidden', type=float, default=None)
    p.add_argument('--theta2-out', type=float, default=None)
    p.add_argument('--pe-steps', type=int, default=5)
    p.add_argument('--rho-embed', type=float, default=1.0)
    p.add_argument('--rho-hidden', type=float, default=3.0)
    p.add_argument('--rho-out', type=float, default=10.0)

    p.add_argument('--prompt', default='To be, or not to be')
    p.add_argument('--sample-tokens', type=int, default=400)
    p.add_argument('--temperature', type=float, default=0.9)
    p.add_argument('--top-k', type=int, default=40)
    p.add_argument('--skip-sample', action='store_true')
    p.add_argument('--no-save', action='store_true')
    return p



def main():
    args = make_parser().parse_args()
    train(args) if args.mode == 'train' else sample(args)


if __name__ == '__main__':
    main()
