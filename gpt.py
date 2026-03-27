import math
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F

TINY_SHAKESPEARE_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"

__all__ = [
    "TINY_SHAKESPEARE_URL",
    "GPTConfig",
    "GPT",
    "CharDataset",
    "BatchSource",
    "maybe_download_tiny_shakespeare",
]


def rms_norm(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return x * torch.rsqrt(x.square().mean(dim=-1, keepdim=True) + eps)


def rms_ball_proj(x: torch.Tensor) -> torch.Tensor:
    return x / (x.square().mean(dim=-1, keepdim=True)).sqrt().clamp_min(1.0)


class Norm(nn.Module):
    def __init__(self, kind: str = "rmsnorm", eps: float = 1e-6):
        super().__init__()
        if kind not in {"rmsnorm", "rmsball"}:
            raise ValueError(f"invalid norm kind: {kind}")
        self.kind = kind
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return rms_norm(x, self.eps) if self.kind == "rmsnorm" else rms_ball_proj(x)


def rotary_cache(seq_len: int, head_dim: int, base: float = 10000.0):
    if head_dim % 2:
        raise ValueError("head_dim must be even for RoPE")
    half = head_dim // 2
    freq = 1.0 / (base ** (torch.arange(half, dtype=torch.float32) / half))
    freqs = torch.outer(torch.arange(seq_len, dtype=torch.float32), freq)
    return freqs.cos(), freqs.sin()


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    x1, x2 = x[..., ::2], x[..., 1::2]
    return torch.stack((x1 * cos - x2 * sin, x1 * sin + x2 * cos), dim=-1).flatten(-2)


class CausalSelfAttention(nn.Module):
    def __init__(
        self, d_model: int, n_head: int, block_size: int, rope_base: float = 10000.0
    ):
        super().__init__()
        if d_model % n_head:
            raise ValueError("d_model must be divisible by n_head")
        self.n_head = n_head
        self.head_dim = d_model // n_head
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.proj = nn.Linear(d_model, d_model, bias=False)
        cos, sin = rotary_cache(block_size, self.head_dim, rope_base)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, seqlen, d_model = x.shape
        qkv = self.qkv(x).view(bsz, seqlen, 3, self.n_head, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        cos = self.rope_cos[:seqlen].view(1, 1, seqlen, self.head_dim // 2)
        sin = self.rope_sin[:seqlen].view(1, 1, seqlen, self.head_dim // 2)
        q = rms_ball_proj(apply_rope(q, cos, sin))
        k = rms_ball_proj(apply_rope(k, cos, sin))
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.proj(y.transpose(1, 2).contiguous().view(bsz, seqlen, d_model))


class MLP(nn.Module):
    def __init__(self, d_model: int, hidden_dim: int):
        super().__init__()
        self.gate = nn.Linear(d_model, hidden_dim, bias=False)
        self.up = nn.Linear(d_model, hidden_dim, bias=False)
        self.down = nn.Linear(hidden_dim, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(F.silu(self.gate(x)) * self.up(x))


class Block(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_head: int,
        hidden_dim: int,
        block_size: int,
        prenorm: str = "rmsnorm",
        rope_base: float = 10000.0,
    ):
        super().__init__()
        self.norm1 = Norm(prenorm)
        self.attn = CausalSelfAttention(d_model, n_head, block_size, rope_base)
        self.norm2 = Norm(prenorm)
        self.mlp = MLP(d_model, hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        return x + self.mlp(self.norm2(x))


@dataclass
class GPTConfig:
    vocab_size: int
    block_size: int = 256
    n_layer: int = 6
    n_head: int = 6
    d_model: int = 384
    rope_base: float = 10000.0
    prenorm: str = "rmsnorm"

    @property
    def hidden_dim(self) -> int:
        return 64 * math.ceil(((8 * self.d_model) // 3) / 64)


class GPT(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.blocks = nn.ModuleList(
            [
                Block(
                    cfg.d_model,
                    cfg.n_head,
                    cfg.hidden_dim,
                    cfg.block_size,
                    cfg.prenorm,
                    cfg.rope_base,
                )
                for _ in range(cfg.n_layer)
            ]
        )
        self.norm_f = Norm(cfg.prenorm)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
        x = self.tok_emb(idx)
        for block in self.blocks:
            x = block(x)
        logits = self.lm_head(self.norm_f(x))
        loss = (
            None
            if targets is None
            else F.cross_entropy(logits.flatten(0, 1), targets.flatten())
        )
        return logits, loss

    @torch.inference_mode()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int = 40,
    ):
        was_training = self.training
        self.eval()
        for _ in range(max_new_tokens):
            logits, _ = self(idx[:, -self.cfg.block_size :])
            logits = logits[:, -1]
            if temperature != 1.0:
                logits = logits / max(temperature, 1e-6)
            if top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits = logits.masked_fill(logits < v[:, [-1]], float("-inf"))
            idx = torch.cat(
                (idx, torch.multinomial(F.softmax(logits, dim=-1), 1)), dim=1
            )
        self.train(was_training)
        return idx


class CharDataset:
    def __init__(self, path: Path):
        text = path.read_text(encoding="utf-8")
        self.chars = sorted(set(text))
        self.stoi = {ch: i for i, ch in enumerate(self.chars)}
        self.itos = {i: ch for i, ch in enumerate(self.chars)}
        data = torch.tensor([self.stoi[c] for c in text], dtype=torch.long)
        n = int(0.9 * len(data))
        self.train = data[:n].contiguous()
        self.val = data[n:].contiguous()

    def encode(self, s: str) -> list[int]:
        return [self.stoi[c] for c in s]

    def decode(self, ids: Iterable[int]) -> str:
        return "".join(self.itos[int(i)] for i in ids)


class BatchSource:
    def __init__(
        self,
        train: torch.Tensor,
        val: torch.Tensor,
        block_size: int,
        batch_size: int,
        device: torch.device,
    ):
        self.block_size = block_size
        self.batch_size = batch_size
        self.device = device
        self.train = train.to(device, non_blocking=True)
        self.val = val.to(device, non_blocking=True)
        self.offsets = torch.arange(block_size + 1, device=device)

    def get(self, split: str):
        data = self.train if split == "train" else self.val
        ix = torch.randint(
            0, data.numel() - self.block_size, (self.batch_size, 1), device=self.device
        )
        chunk = data[ix + self.offsets]
        return chunk[:, :-1], chunk[:, 1:]


def maybe_download_tiny_shakespeare(path: Path):
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(TINY_SHAKESPEARE_URL, path)
