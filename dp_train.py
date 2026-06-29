#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = ["mlx"]
# ///
# pyright: reportMissingImports=false
"""Data-parallel training with gradient all-reduce (MLX distributed, ring backend).

Each rank holds a full model replica and trains on a distinct data shard. Per step:

  local forward/backward  ->  all-reduce (average) gradients across ranks  ->
  identical optimizer update on every replica.

Replicas start identical (shared seed) and apply the same averaged gradient, so they
stay in lock-step — the result equals single-device training at `ranks ×` the batch.
The reported `replica_drift` (max-min of a per-rank parameter checksum) verifies this:
it must stay ~0.

Launch N ranks on one host (ring backend, no MPI needed):

  uv run --with mlx mlx.launch --backend ring --hosts 127.0.0.1 --repeat-hosts N dp_train.py

Baseline (1 rank) runs without the launcher:  uv run dp_train.py
"""
import argparse
import time

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten, tree_map

from train import GPT, loss_fn, median, rand_batch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vocab", type=int, default=8192)
    ap.add_argument("--dim", type=int, default=384)
    ap.add_argument("--heads", type=int, default=6)
    ap.add_argument("--layers", type=int, default=6)
    ap.add_argument("--batch", type=int, default=8, help="per-rank micro-batch")
    ap.add_argument("--seq", type=int, default=256)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--warmup", type=int, default=5)
    args = ap.parse_args()

    world = mx.distributed.init(backend="ring")
    rank, size = world.rank(), world.size()

    mx.random.seed(0)  # identical weights on every replica
    model = GPT(args.vocab, args.dim, args.heads, args.layers, args.seq)
    opt = optim.AdamW(learning_rate=3e-4)
    lg = nn.value_and_grad(model, loss_fn)
    mx.random.seed(1000 + rank)  # distinct data shard per rank

    def step():
        x, y = rand_batch(args.batch, args.seq, args.vocab)
        loss, grads = lg(model, x, y)
        if size > 1:
            grads = tree_map(lambda g: mx.distributed.all_sum(g) / size, grads)  # all-reduce mean
            loss = mx.distributed.all_sum(loss) / size
        opt.update(model, grads)
        mx.eval(model.parameters(), opt.state, loss)
        return loss.item()

    for _ in range(args.warmup):
        step()
    mx.reset_peak_memory()
    times, last = [], 0.0
    for _ in range(args.steps):
        t0 = time.perf_counter()
        last = step()
        times.append(time.perf_counter() - t0)

    # replica-sync check: every rank's parameter checksum must agree.
    # Sum on-GPU in one deferred op (no per-parameter .item() round-trips).
    chk = mx.sum(mx.stack([p.sum() for _, p in tree_flatten(model.parameters())])).reshape(1)
    gathered = mx.distributed.all_gather(chk) if size > 1 else chk
    mx.eval(gathered)
    drift = (gathered.max() - gathered.min()).item()
    assert drift < 1e-2, f"replicas diverged (drift={drift}); gradient all-reduce is broken"

    step_time = median(times)
    tokens = args.batch * args.seq * size  # aggregate tokens/step across ranks
    if rank == 0:
        print(f"ranks={size}  per_rank_batch={args.batch}  eff_batch={args.batch * size}  "
              f"tok/s={tokens / step_time:.0f}  loss={last:.3f}  "
              f"peak/rank={mx.get_peak_memory() / 1e9:.2f}GB  replica_drift={drift:.1e}")


if __name__ == "__main__":
    main()
