# Algorithm Notes

## Tree construction

At each node, the implementation considers every pair of:

1. A sample in the current node subset.
2. A feature index.

The sample's feature value is used as a candidate split. Values greater than
the split go to the right child; all other values go to the left child.

The candidate score is the weighted Gini objective written in an equivalent
form:

```text
score = node_size
        - sum(class_count_left^2) / left_size
        - sum(class_count_right^2) / right_size
```

Lower scores are better. Empty children receive a deliberately poor score.
A split whose two children are pure has score zero; one pure child alone does
not necessarily produce a zero score.

## CPU implementation

The sequential implementation evaluates candidates in nested loops over the
node samples and features. It is the correctness reference and preserves
input order during partitioning. Original labels are retained for predictions;
dense internal class IDs are used for histogram indexing, so arbitrary signed
32-bit labels are supported.

## CUDA implementation

`evaluate_splits_kernel` maps one `(sample, feature)` candidate to each CUDA
thread. Each thread computes class histograms for its candidate and writes one
score to the output array. The host selects the minimum score.

`partition_data_kernel` maps one sample to each CUDA thread. `atomicAdd`
counters reserve positions in the left and right output arrays. Since atomic
writes do not preserve input order, the host sorts each partition back into
the original node order before recursive calls.

## Complexity

For a node with `s` samples and `m` features, candidate evaluation performs
`s * m` candidates, each scanning `s` samples. The split search therefore has
quadratic work in `s` for a fixed feature count. CUDA parallelizes the outer
candidate dimension, while recursive tree construction and host-side score
selection remain sequential.

## Labels

Original labels are retained for leaf predictions. Both implementations use
dense internal IDs in `[0, number_of_classes)` for safe histogram indexing.
CUDA supports up to 32 classes through `MAX_CLASSES`.

## Stopping rules

Leaves predict the majority original label in their node. A node stops when
all labels agree, it has fewer than `MIN_SAMPLES_SPLIT` samples, the optional
`MAX_DEPTH` is reached, no valid split exists, or the best split does not
improve the parent impurity by at least `MIN_IMPURITY_DECREASE`. A split must
also leave at least `MIN_SAMPLES_LEAF` sample in each child. These controls
are compile-time constants near the top of each implementation; the defaults
are `MIN_SAMPLES_SPLIT = 2`, `MIN_SAMPLES_LEAF = 1`, unlimited depth, and no
additional impurity threshold.