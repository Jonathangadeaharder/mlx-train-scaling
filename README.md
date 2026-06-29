# mlx-train-scaling

A minimal transformer **training loop** on Apple Silicon (MLX), instrumented to show how the
levers a training-pipeline engineer pulls — batch size, sequence length, gradient accumulation,
and **data parallelism (gradient all-reduce)** — trade **throughput against memory**.

> **Headline:** gradient accumulation grows the effective batch **4× at +2–7% memory**, where
> a 4× *real* batch costs ~linear memory. But in a lazy framework the win only materializes if
> you realize gradients each micro-step — one misplaced `eval` cost **+43% peak memory**.
> Full analysis: [docs/findings.md](docs/findings.md).

| batch | seq | accum | eff. batch | tok/s | peak GB |
|-------|-----|-------|------------|-------|---------|
| 4 | 256 | 1 | 4 | 15094 | 1.06 |
| 4 | 256 | 4 | **16** | 21986 | **1.13** (+7%) |
| 16 | 256 | 1 | 16 | 43805 | **2.51** (real batch ⇒ +137%) |

## What it does

- Trains a small decoder-only transformer (real forward + backward + AdamW step) on synthetic
  batches — the study is about **systems scaling**, not task accuracy.
- Sweeps batch size × sequence length × gradient-accumulation steps; reports **training tok/s**
  (median, post-warmup) and **peak memory**.
- Ships a **correctness check** that overfits a fixed batch (loss 5.71 → 0.000) so the loop is
  proven to learn, not merely run.

## Why it's written carefully

- **`mx.eval` per micro-step** during accumulation — without it, MLX's lazy graph keeps every
  micro-batch's activations alive and accumulation *raises* memory. This is the core finding.
- **Per-shape warmup discarded** + **median of N steps** — MLX JIT-compiles per tensor shape.

## Usage

```bash
# single-process scaling sweep
uv run train.py --batch-sizes 4 8 16 --seq-lens 256 512 --accum-steps 1 4
uv run train.py --dim 512 --layers 8 --heads 8    # bigger model

# data-parallel: N ranks on one host (MLX ring backend, no MPI)
uv run dp_train.py                                                          # 1-rank baseline
uv run --with mlx mlx.launch --backend ring --hosts 127.0.0.1 --repeat-hosts 4 dp_train.py
```

## Requirements

- macOS, Apple Silicon
- [`uv`](https://docs.astral.sh/uv/) — handles Python + MLX; nothing else to install

## Data parallelism

`dp_train.py` is real data-parallel training on MLX's distributed ring backend — full model
replica per rank, **gradient all-reduce** each step, replicas kept bit-identical (asserted via
`replica_drift`). On this single-GPU machine it demonstrates **correctness**, not speedup:
aggregate throughput falls 1→2 ranks because the ranks time-share one device (same lesson as
[par-vs-batch-bench](https://github.com/Jonathangadeaharder/par-vs-batch-bench) — replication
loses to a bigger batch on one device). The code path is the one that scales on multiple
physical devices. See [docs/findings.md](docs/findings.md#data-parallel-across-processes-gradient-all-reduce).
