import argparse
import math
import time
from dataclasses import asdict
from pathlib import Path

import torch

from gpt import (
    GPT,
    GPTConfig,
    CharDataset,
    BatchSource,
    maybe_download_tiny_shakespeare,
)
from scion import (
    ScionC,
    SpectralLMO,
    ColNormLMO,
    RowNormLMO,
    init_colnorm_,
    init_rownorm_,
    init_semiorthogonal_,
)


def sync_now(device: torch.device) -> float:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return time.perf_counter()


def lr_at_step(
    step: int,
    max_steps: int,
    lr: float,
    min_lr: float,
    warmup_steps: int,
    decay_steps: int,
) -> float:
    if warmup_steps > 0 and step < warmup_steps:
        return lr * (step + 1) / warmup_steps
    decay_steps = min(max(decay_steps, 1), max_steps)
    stable_end = max_steps - decay_steps
    if step < stable_end:
        return lr
    t = min(step - stable_end, decay_steps)
    return lr + (min_lr - lr) * (t / decay_steps)


@torch.inference_mode()
def estimate_loss(
    model: GPT, source: BatchSource, eval_iters: int, amp_dtype: torch.dtype | None
):
    out = {}
    was_training = model.training
    model.eval()
    ctx = (
        torch.autocast(device_type="cuda", dtype=amp_dtype)
        if amp_dtype is not None
        else torch.no_grad()
    )
    with ctx:
        for split in ("train", "val"):
            losses = torch.empty(eval_iters)
            for i in range(eval_iters):
                xb, yb = source.get(split)
                _, loss = model(xb, yb)
                losses[i] = loss.detach().float().cpu()
            out[split] = losses.mean().item()
    model.train(was_training)
    return out


@torch.no_grad()
def init_gpt_scion_(model: GPT):
    init_colnorm_(model.tok_emb.weight)
    for block in model.blocks:
        init_semiorthogonal_(block.attn.qkv.weight)
        init_semiorthogonal_(block.attn.proj.weight)
        init_semiorthogonal_(block.mlp.gate.weight)
        init_semiorthogonal_(block.mlp.up.weight)
        init_semiorthogonal_(block.mlp.down.weight)
    init_rownorm_(model.lm_head.weight)


@torch.no_grad()
def build_scionc(model: GPT, args, device: torch.device) -> ScionC:
    work_dtype = torch.bfloat16 if device.type == "cuda" else None
    lmo_embed = ColNormLMO(radius=args.rho_hidden)
    lmo_hidden = SpectralLMO(
        radius=args.rho_hidden, steps=args.pe_steps, work_dtype=work_dtype
    )
    lmo_out = RowNormLMO(radius=args.rho_out)

    embed, hidden, out = [], [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if name == "tok_emb.weight":
            embed.append(p)
        elif name == "lm_head.weight":
            out.append(p)
        else:
            hidden.append(p)

    groups = []
    if embed:
        groups.append({"params": embed, "dir_fn": lmo_embed})
    if hidden:
        groups.append({"params": hidden, "dir_fn": lmo_hidden})
    if out:
        groups.append({"params": out, "dir_fn": lmo_out})

    return ScionC(
        groups, lr=args.lr, beta2=args.beta2, phi=args.phi, eta=args.eta, cwd=args.cwd
    )


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


def load_checkpoint(path: Path, device: torch.device):
    ckpt = torch.load(path, map_location=device)
    chars = ckpt["chars"]
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}
    model = GPT(GPTConfig(**ckpt["model_cfg"])).to(device)
    model.load_state_dict(ckpt["model"])
    return ckpt, model, chars, stoi, itos


def maybe_compile(
    model: GPT,
    source: BatchSource,
    args,
    amp_dtype: torch.dtype | None,
    device: torch.device,
):
    compile_seconds = 0.0
    compiled = model
    if args.compile and hasattr(torch, "compile"):
        compiled = torch.compile(model)
        xb, yb = source.get("train")
        t0 = sync_now(device)
        compiled.zero_grad(set_to_none=True)
        ctx = (
            torch.autocast(device_type="cuda", dtype=amp_dtype)
            if amp_dtype is not None
            else torch.enable_grad()
        )
        with ctx:
            _, loss = compiled(xb, yb)
        loss.backward()
        compiled.zero_grad(set_to_none=True)
        compile_seconds = sync_now(device) - t0
    return compiled, compile_seconds


def train(args):
    torch.manual_seed(args.seed)
    torch.set_float32_matmul_precision("high")
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    amp_dtype = (
        torch.bfloat16
        if device.type == "cuda" and torch.cuda.is_bf16_supported()
        else None
    )

    data_path = Path(args.data_path)
    maybe_download_tiny_shakespeare(data_path)
    dataset = CharDataset(data_path)

    cfg = GPTConfig(
        vocab_size=len(dataset.chars),
        block_size=args.block_size,
        n_layer=args.n_layer,
        n_head=args.n_head,
        d_model=args.d_model,
        rope_base=args.rope_base,
        prenorm=args.prenorm,
    )
    raw_model = GPT(cfg).to(device)
    init_gpt_scion_(raw_model)
    source = BatchSource(
        dataset.train, dataset.val, args.block_size, args.batch_size, device
    )
    opt = build_scionc(raw_model, args, device)
    model, compile_seconds = maybe_compile(raw_model, source, args, amp_dtype, device)

    if compile_seconds > 0.0:
        print(f"compile_seconds {compile_seconds:.3f}")

    micro_tokens = args.batch_size * args.block_size
    effective_tokens = micro_tokens * args.grad_accum
    train_start = sync_now(device)
    total_opt_steps = 0
    best_val = float("inf")
    last_losses = {"train": float("nan"), "val": float("nan")}
    initial_val = None
    max_val = float("-inf")
    diverged = False
    diverge_reason = ""

    warmup_steps = (
        args.warmup_iters
        if args.warmup_iters >= 0
        else int(round(args.warmup_frac * args.max_iters))
    )
    decay_steps = int(round(args.decay_frac * args.max_iters))

    for step in range(args.max_iters):
        lr = lr_at_step(
            step, args.max_iters, args.lr, args.min_lr, warmup_steps, decay_steps
        )
        for group in opt.param_groups:
            group["lr"] = lr

        if step % args.eval_interval == 0 or step == args.max_iters - 1:
            losses = estimate_loss(model, source, args.eval_iters, amp_dtype)
            last_losses = losses
            train_loss = losses["train"]
            val_loss = losses["val"]
            if not (math.isfinite(train_loss) and math.isfinite(val_loss)):
                diverged = True
                diverge_reason = "nonfinite_eval_loss"
            else:
                if initial_val is None:
                    initial_val = val_loss
                max_val = max(max_val, val_loss)
                best_val = min(best_val, val_loss)
                if (
                    step > 0
                    and initial_val is not None
                    and val_loss > initial_val * args.diverge_mult
                ):
                    diverged = True
                    diverge_reason = (
                        f"val_loss_exceeded_{args.diverge_mult:.2f}x_initial"
                    )
            train_elapsed = max(sync_now(device) - train_start, 1e-9)
            tok_per_s = (total_opt_steps * effective_tokens) / train_elapsed
            print(
                f"step {step:5d} | lr {lr:.3e} | "
                f"train {train_loss:.4f} | val {val_loss:.4f} | best_val {best_val:.4f} | "
                f"train_seconds {train_elapsed:.3f} | tok/s {tok_per_s:.0f}"
            )
            if not args.no_save:
                save_checkpoint(Path(args.out_path), raw_model, dataset, args)
            if diverged:
                print(f"diverged {diverge_reason}")
                break

        opt.zero_grad(set_to_none=True)
        for _ in range(args.grad_accum):
            xb, yb = source.get("train")
            ctx = (
                torch.autocast(device_type="cuda", dtype=amp_dtype)
                if amp_dtype is not None
                else torch.enable_grad()
            )
            with ctx:
                _, loss = model(xb, yb)
                loss = loss / args.grad_accum
            if not torch.isfinite(loss.detach()):
                diverged = True
                diverge_reason = "nonfinite_train_loss"
                break
            loss.backward()
        if diverged:
            print(f"diverged {diverge_reason}")
            break
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(raw_model.parameters(), args.grad_clip)
        opt.step()
        total_opt_steps += 1

    if not args.skip_sample and not diverged:
        prompt = args.prompt or "\n"
        x = torch.tensor([dataset.encode(prompt)], dtype=torch.long, device=device)
        y = raw_model.generate(
            x,
            max_new_tokens=args.sample_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
        )
        print("\n--- sample ---\n")
        print(dataset.decode(y[0].tolist()))
    return {
        "best_val": best_val,
        "final_train": last_losses["train"],
        "final_val": last_losses["val"],
        "compile_seconds": compile_seconds,
        "initial_val": initial_val if initial_val is not None else float("nan"),
        "max_val": max_val,
        "diverged": diverged,
        "diverge_reason": diverge_reason,
    }


@torch.inference_mode()
def sample(args):
    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    _, model, _, stoi, itos = load_checkpoint(Path(args.out_path), device)
    prompt = args.prompt or "\n"
    bad = [c for c in prompt if c not in stoi]
    if bad:
        raise ValueError(f"prompt contains unseen chars: {bad}")
    x = torch.tensor([[stoi[c] for c in prompt]], dtype=torch.long, device=device)
    y = model.generate(
        x,
        max_new_tokens=args.sample_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
    )
    print("".join(itos[int(i)] for i in y[0].tolist()))


def make_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["train", "sample"], default="train")
    p.add_argument("--data-path", default="data/tiny_shakespeare.txt")
    p.add_argument("--out-path", default="out/scionc_shakespeare.pt")
    p.add_argument("--device", default="")
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--compile", action="store_true")

    p.add_argument("--block-size", type=int, default=256)
    p.add_argument("--batch-size", type=int, default=64, help="microbatch size")
    p.add_argument("--grad-accum", type=int, default=1)
    p.add_argument("--n-layer", type=int, default=6)
    p.add_argument("--n-head", type=int, default=6)
    p.add_argument("--d-model", type=int, default=384)
    p.add_argument("--rope-base", type=float, default=10000.0)
    p.add_argument("--prenorm", choices=["rmsnorm", "rmsball"], default="rmsnorm")

    p.add_argument("--max-iters", type=int, default=3000)
    p.add_argument("--eval-interval", type=int, default=200)
    p.add_argument("--eval-iters", type=int, default=50)
    p.add_argument("--grad-clip", type=float, default=0.0)
    p.add_argument("--diverge-mult", type=float, default=2.0)

    p.add_argument(
        "--warmup-iters", type=int, default=-1, help="if >=0, overrides warmup-frac"
    )
    p.add_argument("--warmup-frac", type=float, default=0.0)
    p.add_argument("--decay-frac", type=float, default=0.2)

    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--min-lr", type=float, default=0.0)
    p.add_argument("--beta2", type=float, default=0.95)
    p.add_argument("--phi", type=float, default=0.0)
    p.add_argument("--eta", type=float, default=0.0)
    p.add_argument("--cwd", action="store_true")
    p.add_argument("--pe-steps", type=int, default=5)
    p.add_argument("--rho-hidden", type=float, default=50.0)
    p.add_argument("--rho-out", type=float, default=3000.0)

    p.add_argument("--prompt", default="To be, or not to be")
    p.add_argument("--sample-tokens", type=int, default=400)
    p.add_argument("--temperature", type=float, default=0.9)
    p.add_argument("--top-k", type=int, default=40)
    p.add_argument("--skip-sample", action="store_true")
    p.add_argument("--no-save", action="store_true")
    return p


def main():
    args = make_parser().parse_args()
    if args.mode == "train":
        train(args)
    else:
        sample(args)


if __name__ == "__main__":
    main()
