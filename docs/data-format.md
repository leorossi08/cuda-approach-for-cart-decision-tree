# Binary Data Format

The programs intentionally use a simple binary format for fast loading.

## Training file

For `n` samples and `m` features, `train.bin` contains, in order:

1. `n * m` little-endian IEEE-754 64-bit floating-point values.
2. `n` little-endian signed 32-bit integer labels.

Features are row-major:

```text
x[0][0], x[0][1], ..., x[0][m-1],
x[1][0], x[1][1], ..., x[1][m-1], ...
```

Expected file size:

```text
n * m * 8 + n * 4 bytes
```

Labels are signed 32-bit integers and may have arbitrary values. The tree
internally converts them to dense class IDs for histogram indexing, while
preserving the original values for leaf predictions.

## Test file

`test.bin` contains only `n_test * m` little-endian 64-bit doubles in the
same row-major layout.

Expected file size:

```text
n_test * m * 8 bytes
```

The C and CUDA loaders use `fread`, so the files should be generated or
converted on a compatible little-endian machine. The included Python
generator explicitly writes little-endian values.