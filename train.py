#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = ["mlx"]
# ///
# pyright: reportMissingImports=false
"""Transformer training-loop scaling study on Apple Silicon (MLX).

Measures the levers a training-pipeline engineer actually pulls, and how each
trades training throughput against memory:

  - batch size        (more parallelism per step)
  - sequence length   (attention cost grows ~quadratically)
  - gradient accumulation (larger effective batch without the memory of a big one)

Throughput is tokens/second of *training* (forward + backward + optimizer step),
median over N steps after warmup. Memory is peak MLX allocation.

A separate correctness check overfits a single fixed batch and asserts the loss
falls — proving the loop learns, not just runs.

Run:  uv run train.py
"""

import argparse
import time

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_map


# ---------------------------------------------------------------------------
# Model: a small decoder-only transformer (GPT-style)
# ---------------------------------------------------------------------------
class Block(nn.Module):
    def __init__(self, dim: int, heads: int):
        super().__init__()
        self.ln1 = nn.LayerNorm(dim)
        self.attn = nn.MultiHeadAttention(dim, heads)
        self.ln2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, 4 * dim), nn.GELU(), nn.Linear(4 * dim, dim))

    def __call__(self, x, mask):
        h = self.ln1(x)
        x = x + self.attn(h, h, h, mask)
        return x + self.mlp(self.ln2(x))


class GPT(nn.Module):
    def __init__(self, vocab: int, dim: int, heads: int, layers: int, max_len: int):
        super().__init__()
        self.tok = nn.Embedding(vocab, dim)
        self.pos = nn.Embedding(max_len, dim)
        self.blocks = [Block(dim, heads) for _ in range(layers)]
        self.ln_f = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, vocab)

    def __call__(self, idx):
        _, L = idx.shape
        x = self.tok(idx) + self.pos(mx.arange(L))
        mask = mx.triu(mx.full((L, L), -1e9), k=1)  # causal (additive)
        for blk in self.blocks:
            x = blk(x, mask)
        return self.head(self.ln_f(x))


def loss_fn(model, x, y):
    logits = model(x)
    vocab = logits.shape[-1]
    return nn.losses.cross_entropy(logits.reshape(-1, vocab), y.reshape(-1), reduction="mean")


# ---------------------------------------------------------------------------
def median(xs: list[float]) -> float:
    xs = sorted(xs)
    n = len(xs)
    return xs[n // 2] if n % 2 else (xs[n // 2 - 1] + xs[n // 2]) / 2


def rand_batch(B: int, L: int, vocab: int):
    d = mx.random.randint(0, vocab, (B, L + 1))
    return d[:, :-1], d[:, 1:]


def train_steps(model, opt, B, L, vocab, accum, steps, warmup):
    """Run accum-micro-batch steps; return (tokens/s median, peak GB)."""
    lg = nn.value_and_grad(model, loss_fn)

    def one_step():
        grads = None
        for _ in range(accum):
            x, y = rand_batch(B, L, vocab)
            _, g = lg(model, x, y)
            grads = g if grads is None else tree_map(lambda a, b: a + b, grads, g)
            mx.eval(grads)  # realize+free this microbatch's graph; else lazy MLX holds
            # every microbatch's activations live, defeating accumulation's memory win
        if accum > 1:
            grads = tree_map(lambda a: a / accum, grads)
        opt.update(model, grads)
        mx.eval(model.parameters(), opt.state)

    for _ in range(warmup):  # discard: MLX JIT-compiles per shape
        one_step()
    mx.reset_peak_memory()
    times = []
    for _ in range(steps):
        t0 = time.perf_counter()
        one_step()
        times.append(time.perf_counter() - t0)
    tokens_per_step = B * L * accum
    return tokens_per_step / median(times), mx.get_peak_memory() / 1e9


def correctness_check(vocab=256, dim=128, heads=4, layers=2, B=8, L=32, steps=120):
    """Overfit one fixed batch; loss must fall substantially."""
    model = GPT(vocab, dim, heads, layers, L)
    opt = optim.AdamW(learning_rate=3e-3)
    lg = nn.value_and_grad(model, loss_fn)
    x, y = rand_batch(B, L, vocab)
    first = last = 0.0
    for i in range(steps):
        loss, g = lg(model, x, y)
        opt.update(model, g)
        mx.eval(model.parameters(), opt.state, loss)
        if i == 0:
            first = loss.item()
        last = loss.item()
    return first, last


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vocab", type=int, default=8192)
    ap.add_argument("--dim", type=int, default=384)
    ap.add_argument("--heads", type=int, default=6)
    ap.add_argument("--layers", type=int, default=6)
    ap.add_argument("--max-len", type=int, default=1024)
    ap.add_argument("--batch-sizes", type=int, nargs="+", default=[4, 8, 16])
    ap.add_argument("--seq-lens", type=int, nargs="+", default=[256, 512])
    ap.add_argument("--accum-steps", type=int, nargs="+", default=[1, 4])
    ap.add_argument("--steps", type=int, default=15)
    ap.add_argument("--warmup", type=int, default=3)
    args = ap.parse_args()

    n_params = sum(p.size for _, p in _flat_params(GPT(args.vocab, args.dim, args.heads, args.layers, args.max_len)))
    print(f"model: dim={args.dim} layers={args.layers} heads={args.heads} ~{n_params/1e6:.1f}M params")

    first, last = correctness_check()
    print(f"correctness: overfit loss {first:.3f} -> {last:.3f}  ({'PASS' if last < first * 0.5 else 'FAIL'})\n")

    print(f"{'batch':<7}{'seqlen':<8}{'accum':<7}{'eff_batch':<11}{'tok/s':<10}{'peak GB':<9}")
    print("-" * 52)
    for L in args.seq_lens:
        for B in args.batch_sizes:
            for accum in args.accum_steps:
                model = GPT(args.vocab, args.dim, args.heads, args.layers, args.max_len)
                opt = optim.AdamW(learning_rate=3e-4)
                tps, peak = train_steps(model, opt, B, L, args.vocab, accum, args.steps, args.warmup)
                print(f"{B:<7}{L:<8}{accum:<7}{B*accum:<11}{tps:<10.0f}{peak:<9.2f}")
                del model, opt
                mx.clear_cache()


def _flat_params(model):
    from mlx.utils import tree_flatten
    return tree_flatten(model.parameters())


if __name__ == "__main__":
    main()
