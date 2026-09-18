# CPU and CUDA Implementation Comparison

Both programs build the same binary classification tree. They use the same
candidate splits, the same `value > split` routing rule, and the same weighted
Gini objective. The CPU implementation is the reference implementation; the
CUDA implementation accelerates the most expensive repeated split evaluation.

## Shared tree algorithm

At each node, both programs:

1. Keep the indices of samples belonging to the node.
2. Test every pair of node sample and feature as a candidate split.
3. Send values greater than the candidate threshold right, and all other
   values left.
4. Compute the weighted Gini score for both children.
5. Select the lowest-scoring valid candidate.
6. Partition the node and recurse into the two children.

The score is:

```text
score = node_size
        - sum(class_count_left^2) / left_size
        - sum(class_count_right^2) / right_size
```

Both implementations also share these rules:

- Leaves predict the majority original label in the node.
- Labels are converted to dense internal IDs for histogram indexing.
- Empty or non-improving splits are rejected.
- A split must satisfy the minimum child-size rule.
- Original labels are retained for predictions.

## High-level comparison

| Area | Sequential implementation | CUDA implementation |
|---|---|---|
| Source | `src/cart_sequential.c` | `src/cart_cuda.cu` |
| Candidate evaluation | Nested CPU loops | One CUDA thread per candidate |
| Candidate count | `samples * features` | `samples * features` |
| Work per candidate | Scans all node samples | Scans all node samples on the GPU |
| Score selection | CPU loop | Scores copied to host, then CPU selects minimum |
| Partitioning | CPU loop, naturally ordered | One CUDA thread per sample with atomic counters |
| Partition order | Preserved directly | Restored on host after copying |
| Recursion | CPU recursive calls | Host recursive calls; each node launches GPU work |
| Prediction | CPU traversal | CPU traversal over the host tree |
| Label histograms | Dense internal IDs | Dense internal IDs, limited to 32 classes |

## Where CUDA is used

### Candidate split evaluation

`evaluate_splits_kernel` performs the main parallel operation. Each thread
receives one `(sample, feature)` pair:

```text
thread -> candidate threshold
candidate -> scan node samples
scan -> build left/right class histograms
histograms -> write one Gini score
```

This is a good CUDA target because candidates are independent. Every candidate
reads the same node data, computes its own histograms, and writes to a unique
score slot. There is no need for threads evaluating different candidates to
communicate with one another.

The CPU version performs the same work in nested loops, so it evaluates only
one candidate at a time. For a node with `s` samples and `m` features, both
implementations do `s * m` candidate evaluations and each candidate scans `s`
samples. The total split-search work is therefore quadratic in `s` for a fixed
feature count; CUDA parallelizes the outer candidate dimension.

### Data partitioning

`partition_data_kernel` assigns one thread to each sample in the current node.
The thread evaluates the chosen split and reserves an output position with
`atomicAdd` in either the left or right partition.

Atomic counters are used because many threads may append to the same child
array concurrently. The resulting order is nondeterministic, so the host
sorts each partition according to the original node order. This preserves the
same recursive tie-breaking behavior as the CPU implementation.

## What remains on the CPU and why

### Choosing the best score

The GPU writes one score per candidate, then copies the score array to the
host. The host scans it to select the minimum. This keeps the implementation
simple and matches the CPU tie-breaking rule: equal scores keep the first
candidate encountered.

A GPU reduction could make this step faster for very large nodes, but it would
add reduction and tie-breaking logic. The current host scan is small compared
with the repeated per-candidate histogram work.

### Tree recursion

The tree is built recursively on the host. Each node has a different subset
size and launches new GPU work, so the recursion has irregular control flow.
Keeping recursion on the host avoids device-side dynamic allocation and makes
the tree structure easy to manage with normal pointers.

### Prediction

Prediction is a tree traversal for each test sample. It performs only one
comparison per tree level, so it has much less work than training. The CUDA
program therefore keeps the tree and prediction traversal on the CPU.

### Host-side partition ordering

The GPU partition kernel is used for parallel classification of node samples,
but the host restores the original order afterward. This is required because
atomic appends do not preserve input order, and preserving order makes CPU and
CUDA trees reproducible when candidate scores tie.

## Why not use CUDA everywhere?

CUDA is most useful when there is enough regular, independent work to offset
kernel-launch and memory-transfer overhead. Candidate scoring has exactly that
shape: many independent candidates perform the same scan and histogram
operation.

Tree recursion, score selection, and prediction have less parallel work or
more irregular control flow. Moving those parts to the GPU would increase
complexity and synchronization without necessarily improving total runtime.

The CUDA implementation still synchronizes and transfers data at each node.
This means it may be slower than the CPU implementation for small datasets or
small node subsets. Its advantage is expected on larger nodes where many
candidate evaluations can run concurrently.

## Correctness and implementation differences

The implementations intentionally differ in execution strategy, but their
observable tree behavior is kept aligned:

- CPU candidate order is sample-major, then feature-major.
- CUDA stores candidates in the same sample-major, feature-major order.
- The host selects the first minimum score in both implementations.
- CPU partitioning preserves node order naturally.
- CUDA restores node order after atomic partitioning.
- Both use original labels for majority leaves and predictions.
- Both use dense labels for class-count indexing.

The CUDA histogram arrays use `MAX_CLASSES`, currently 32. Datasets with more
than 32 distinct classes are rejected by the CUDA program.

## Summary

The CPU implementation provides a straightforward correctness reference and
handles the complete tree-building workflow sequentially. CUDA is used for the
two most expensive regular operations: evaluating all candidate splits and
classifying samples into the chosen child partitions. The remaining work stays
on the host because it is recursive, order-sensitive, or too small to justify
additional GPU coordination.
