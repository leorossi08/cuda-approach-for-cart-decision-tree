# CUDA CART Decision Tree

CUDA and sequential implementations of a CART-style classification tree,
with reproducible data generation, correctness comparison, and performance
benchmarks.

The CUDA implementation evaluates candidate splits in parallel on the GPU.
The sequential implementation is a CPU reference with the same weighted Gini
objective. Both implementations use the same binary input format and produce
the same predictions on the comparison tests.

## Requirements

### Software

- Linux
- GCC
- GNU Make
- Python 3.10 or newer
- NVIDIA CUDA Toolkit with `nvcc`
- NVIDIA driver compatible with the installed CUDA toolkit

No third-party Python packages are needed. See
[requirements-system.txt](requirements-system.txt) and
[requirements.txt](requirements.txt).

### GPU

The tested configuration was:

- NVIDIA GeForce MX110
- Compute capability 5.0
- CUDA toolkit 12.4
- NVIDIA driver 580.178.04

For another GPU, set its architecture when building, for example:

```bash
make CUDA_ARCH=sm_86
```

Find the installed GPU and driver with:

```bash
nvidia-smi
```

## Build From Scratch

From the repository root:

```bash
make
```

This creates:

- `build/cart_cuda`
- `build/cart_sequential`

The default architecture is `sm_50`, matching the tested MX110.

## Run A Correctness Test

Generate a 1,000-sample training set and 200-sample test set, run both
implementations, and compare all predictions:

```bash
make test
```

Expected final line:

```text
PASS: CUDA and sequential predictions match
```

Generated data is placed in `data/`; result files are placed in `results/`.
Both are ignored by Git.

## Run The Benchmark

Run the standard size/class sweep:

```bash
make benchmark
```

This tests training sizes `500,1000,2000`, class counts `2,3,5`, and three
runs per scenario. The output is a table containing tree time, end-to-end
wall time, speedup, prediction agreement, and synthetic-rule accuracy.

Run the longer GPU-oriented suite:

```bash
make heavy
```

The heavy suite tests sizes `2000,4000,8000`, classes `2,5,10`, 500 test
samples, and two runs per scenario. The full measured report is in
[docs/benchmark-results.md](docs/benchmark-results.md).

Custom benchmark example:

```bash
python3 scripts/benchmark_compare.py \
  --train-sizes 1000,4000,8000 \
  --class-counts 2,5,10 \
  --test-size 500 \
  --runs 5
```

## Generate Data Manually

The generator supports 2 to 32 dense classes:

```bash
python3 scripts/generate_data.py \
  --train-size 2000 \
  --test-size 500 \
  --features 4 \
  --classes 5 \
  --train-file data/train.bin \
  --test-file data/test.bin
```

The training file contains all feature doubles followed by all integer
labels. The test file contains only feature doubles. See
[docs/data-format.md](docs/data-format.md).

## Run Programs Directly

The executable arguments are:

```text
<n_train> <m_features> <n_test> <train_file> <test_file> <output_file>
```

Example:

```bash
./build/cart_cuda 1000 4 200 data/train.bin data/test.bin results/cuda.txt
./build/cart_sequential 1000 4 200 data/train.bin data/test.bin results/sequential.txt
python3 scripts/compare_results.py results/sequential.txt results/cuda.txt
```

## Repository Layout

```text
src/
  cart_cuda.cu          CUDA implementation
  cart_sequential.c     CPU reference implementation
scripts/
  generate_data.py      deterministic binary data generator
  compare_results.py    exact prediction comparison
  benchmark_compare.py  readable multi-scenario benchmark
  tests_heavy.py        longer GPU-focused benchmark
docs/
  algorithm.md          implementation and correctness details
  data-format.md        binary file specification
  benchmark-results.md  measured benchmark report
data/                    generated files, ignored by Git
results/                 generated reports, ignored by Git
Makefile                build and test commands
requirements.txt         Python dependency declaration
```

## Correctness Notes

Both implementations minimize the weighted Gini objective:

```text
score = samples - left_purity - right_purity
```

CUDA partitioning uses atomic counters for parallel writes, then restores the
input order on the host. This makes recursive tie-breaking deterministic and
allows exact prediction comparison with the sequential implementation.

## Clean Up

```bash
make clean
```

This removes compiled binaries, generated binary datasets, and result files.

## Measured Benchmark Results

The following results were measured with the README commands on an NVIDIA
GeForce MX110 with compute capability 5.0, driver 580.178.04, and 2 GiB of
GPU memory. Every scenario produced identical CPU and CUDA predictions.

### Standard Benchmark

The standard benchmark uses three runs per scenario and 200 test samples.

| Train | Classes | CPU tree ms | CUDA tree ms | Tree speedup | CPU wall ms | CUDA wall ms | Wall speedup | Match | CPU accuracy | CUDA accuracy |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 500 | 2 | 7.38 | 3.13 | 2.35x | 8.45 | 73.28 | 0.12x | 200/200 | 96.5% | 96.5% |
| 500 | 3 | 8.28 | 4.67 | 1.77x | 9.33 | 90.88 | 0.10x | 200/200 | 94.5% | 94.5% |
| 500 | 5 | 7.73 | 6.90 | 1.12x | 8.77 | 87.31 | 0.10x | 200/200 | 87.5% | 87.5% |
| 1000 | 2 | 29.16 | 5.90 | 4.94x | 30.26 | 85.56 | 0.35x | 200/200 | 98.5% | 98.5% |
| 1000 | 3 | 36.12 | 9.77 | 3.70x | 37.22 | 76.05 | 0.49x | 200/200 | 97.0% | 97.0% |
| 1000 | 5 | 32.47 | 12.63 | 2.57x | 33.57 | 82.66 | 0.41x | 200/200 | 93.5% | 93.5% |
| 2000 | 2 | 137.35 | 14.97 | 9.18x | 138.56 | 90.95 | 1.52x | 200/200 | 98.0% | 98.0% |
| 2000 | 3 | 147.28 | 21.43 | 6.87x | 148.56 | 90.86 | 1.64x | 200/200 | 96.5% | 96.5% |
| 2000 | 5 | 228.07 | 35.93 | 6.35x | 230.22 | 140.47 | 1.64x | 200/200 | 97.5% | 97.5% |

### Heavy Benchmark

The heavy benchmark uses two runs per scenario, 500 test samples, training
sizes of 2000, 4000, and 8000, and class counts of 2, 5, and 10.

| Train | Classes | CPU tree ms | CUDA tree ms | Tree speedup | CPU wall ms | CUDA wall ms | Wall speedup | Match | CPU accuracy | CUDA accuracy |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2000 | 2 | 277.18 | 20.80 | 13.33x | 279.34 | 134.10 | 2.08x | 500/500 | 97.6% | 97.6% |
| 2000 | 5 | 132.61 | 26.40 | 5.02x | 134.08 | 99.38 | 1.35x | 500/500 | 96.6% | 96.6% |
| 2000 | 10 | 129.91 | 32.90 | 3.95x | 131.20 | 100.15 | 1.31x | 500/500 | 89.2% | 89.2% |
| 4000 | 2 | 570.38 | 52.05 | 10.96x | 573.79 | 154.53 | 3.71x | 500/500 | 99.0% | 99.0% |
| 4000 | 5 | 903.39 | 72.50 | 12.46x | 911.52 | 159.56 | 5.71x | 500/500 | 94.0% | 94.0% |
| 4000 | 10 | 525.51 | 74.80 | 7.03x | 530.44 | 148.98 | 3.56x | 500/500 | 91.2% | 91.2% |
| 8000 | 2 | 3273.59 | 167.45 | 19.55x | 3282.00 | 238.76 | 13.75x | 500/500 | 99.4% | 99.4% |
| 8000 | 5 | 2984.01 | 202.05 | 14.77x | 2993.56 | 296.27 | 10.10x | 500/500 | 98.0% | 98.0% |
| 8000 | 10 | 2843.62 | 229.50 | 12.39x | 2850.96 | 308.33 | 9.25x | 500/500 | 94.0% | 94.0% |

### Results Summary

- `make test`: CPU/CUDA predictions matched `200/200`.
- Standard benchmark: all 9 scenarios matched exactly.
- Heavy benchmark: all 9 scenarios matched exactly.
- Heavy CUDA tree-construction speedup: `3.95x` to `19.55x`.
- Heavy complete wall-time speedup: `1.31x` to `13.75x`.
- CPU and CUDA accuracy was identical in every scenario.