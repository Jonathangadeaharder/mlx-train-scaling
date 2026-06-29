# Findings: training-loop scaling levers on Apple Silicon

**TL;DR** — For transformer training, **gradient accumulation is the memory-cheap way to grow
the effective batch**: 4× effective batch cost **+2–7% memory**, where growing the *real* batch
4× costs **~linear memory** (1.06 → 2.51 GB). But this only holds if you **realize gradients
each micro-step** — in a lazy framework, naive accumulation holds every micro-batch's
activation graph live and the memory win evaporates (measured **−43%** peak from one correctly
placed `mx.eval`). Sequence length is the expensive axis: doubling it ~doubles memory.

## Setup

- **Hardware:** Apple M5 Pro, 64 GB unified memory, macOS 26.5.1
- **Stack:** MLX 0.31.x, Python 3.13 (via `uv`)
- **Model:** decoder-only transformer, dim 384 / 6 layers / 6 heads, ~17.3 M params
- **Workload:** synthetic token batches (the study is about systems scaling, not task accuracy)
- **Metric:** training tokens/s (forward + backward + optimizer step), median over 12 steps
  after 3 warmup steps; peak memory via `mx.get_peak_memory()`
- **Correctness:** a separate check overfits one fixed batch — loss **5.71 → 0.000** — proving
  the loop actually learns, not just executes.

## Results

| batch | seq len | accum | eff. batch | tok/s  | peak GB |
|-------|---------|-------|------------|--------|---------|
| 4     | 256     | 1     | 4          | 15094  | 1.06    |
| 4     | 256     | 4     | 16         | 21986  | 1.13    |
| 8     | 256     | 1     | 8          | 22194  | 1.76    |
| 8     | 256     | 4     | 32         | 45965  | 1.84    |
| 16    | 256     | 1     | 16         | 43805  | 2.51    |
| 16    | 256     | 4     | 64         | 49726  | 2.51    |
| 4     | 512     | 1     | 4          | 38501  | 1.99    |
| 4     | 512     | 4     | 16         | 44350  | 2.06    |
| 8     | 512     | 1     | 8          | 33782  | 2.75    |
| 16    | 512     | 1     | 16         | 26727  | 4.12    |
| 16    | 512     | 4     | 64         | 27621  | 4.19    |

## Interpretation

**1. Gradient accumulation buys effective batch at near-constant memory.**
Holding micro-batch fixed and raising `accum` 1 → 4 (4× effective batch) moves peak memory only
+2–7% (e.g. B4/256: 1.06 → 1.13 GB; B16/256: 2.51 → 2.51 GB). Activation memory is set by the
*micro-batch*, not the effective batch — accumulation sums gradients across micro-steps, and
gradients are model-sized, not batch-sized.

**2. Real batch size grows memory ~linearly.**
B4 → B8 → B16 at seq 256: 1.06 → 1.76 → 2.51 GB. So to hit a target effective batch, raising
the *real* batch is the expensive path; accumulation is the cheap one. The two also compose —
pick the largest micro-batch that fits, then use accumulation for the rest.

**3. Sequence length is the costly axis.**
256 → 512 at fixed batch nearly doubles memory (B4: 1.06 → 1.99; B16: 2.51 → 4.12 GB) —
activations scale with length and attention scales with its square. Long-context training hits
the memory wall here, not at large batch.

**4. The lazy-evaluation trap (why the loop is written the way it is).**
MLX is lazy: an op builds a graph, executed only on `eval`. A naive accumulation loop that
evals once per optimizer step keeps **all** micro-batches' activation graphs alive
simultaneously — so accumulation *raises* memory instead of holding it flat. Measured at
B4/256 accum 4: **2.00 GB naive vs 1.13 GB** with `mx.eval(grads)` after each micro-step
(−43%). The fix is one line, and it is the difference between accumulation working and not.

**5. Throughput rises with effective batch until bandwidth-bound, then flattens.**
Gains are clean at small/mid sizes; at seq 512 + large batch the device saturates and
throughput plateaus or regresses, with run-to-run variance from thermal and scheduling. Treat
the high end as "bandwidth-bound, diminishing returns," not as precise rankings.

## Practical guidance

- **Reach for accumulation, not a bigger real batch,** when you need a larger effective batch
  and memory is tight.
- **In a lazy framework, eval gradients every micro-step.** Otherwise accumulation silently
  costs the memory it was meant to save.
- **Budget memory by sequence length first**, then micro-batch; accumulation is nearly free.
- **Stop scaling effective batch once throughput flattens** — past the bandwidth knee you pay
  compute for no tok/s.

## Limitations & next steps

- Single device, single process — this measures the *single-node* training levers. The natural
  extension is **data-parallel across processes with gradient all-reduce** (the distributed
  pattern), which this harness is structured to add.
- Synthetic data and a small model; absolute tok/s is not comparable to production training,
  but the *scaling shapes* (accumulation vs real batch vs seq len) are model-agnostic.
- Greedy/dense attention, no flash-attention kernel, no activation checkpointing — each would
  shift the memory/throughput constants.

## Reproduce

```bash
uv run train.py --batch-sizes 4 8 16 --seq-lens 256 512 --accum-steps 1 4
```
