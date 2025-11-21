# Tensara Evaluation Suites

Consistency evaluation suite for measuring engine benchmark variance. Runs the same CUDA/Triton submissions multiple times and measures variance in runtime results.

## How It Works

- Executes each submission multiple times (default: 10 runs) via Modal endpoint
- Calculates statistics: mean, median, stdev, CV, min/max
- Tracks per-testcase runtime variance
- Randomizes run order (optional seed)

## What It Measures

- Runtime variance (stdev, CV) across runs
- Per-testcase consistency
- Success rate (successful vs failed runs)
- Performance stability indicators
