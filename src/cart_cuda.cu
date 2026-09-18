#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include <algorithm>
#include <unordered_map>

#define CUDA_CHECK(call)                                                     \
do {                                                                         \
    cudaError_t err__ = (call);                                              \
    if (err__ != cudaSuccess) {                                              \
        fprintf(stderr, "CUDA error in %s:%d: %s\n",                       \
                __FILE__, __LINE__, cudaGetErrorString(err__));              \
        exit(EXIT_FAILURE);                                                  \
    }                                                                        \
} while (0)

// The kernel keeps class histograms in registers/local memory.
// Increase this if your dataset has more classes.
#define MAX_CLASSES 32
#define MAX_DEPTH (-1)
#define MIN_SAMPLES_SPLIT 2
#define MIN_SAMPLES_LEAF 1
#define MIN_IMPURITY_DECREASE 0.0

typedef struct btree {
    double data;               // split value for internal nodes
    int num_q;                 // feature index for internal nodes; class for leaves
    struct btree* left;
    struct btree* right;
} btree;

// Evaluate one candidate (sample, feature) per CUDA thread.
// y must contain dense class IDs in [0, noc).
__global__ void evaluate_splits_kernel(
    const double* __restrict__ x,
    int m,
    const int* __restrict__ y,
    const int* __restrict__ nums,
    int sch,
    int noc,
    double* __restrict__ out_ginis) {

    const int candidate_idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_candidates = sch * m;
    if (candidate_idx >= total_candidates) return;

    const int sample_pos = candidate_idx / m;
    const int feature_idx = candidate_idx - sample_pos * m;
    const int sample_idx = nums[sample_pos];
    const double split_val = x[sample_idx * m + feature_idx];

    int left_counts[MAX_CLASSES] = {0};
    int right_counts[MAX_CLASSES] = {0};
    int left_total = 0;
    int right_total = 0;

    for (int i = 0; i < sch; ++i) {
        const int s = nums[i];
        const double v = x[s * m + feature_idx];
        const int cls = y[s];

        if (v > split_val) {
            ++right_total;
            ++right_counts[cls];
        } else {
            ++left_total;
            ++left_counts[cls];
        }
    }

    if (left_total == 0 || right_total == 0) {
        // Invalid split: make it worse than any valid split.
        out_ginis[candidate_idx] = static_cast<double>(sch);
        return;
    }

    long long left_sq_sum = 0;
    long long right_sq_sum = 0;

    for (int c = 0; c < noc; ++c) {
        left_sq_sum += static_cast<long long>(left_counts[c]) * left_counts[c];
        right_sq_sum += static_cast<long long>(right_counts[c]) * right_counts[c];
    }

    // Equivalent to minimizing the weighted Gini impurity:
    //   sch - sum_c(n_Lc^2 / n_L) - sum_c(n_Rc^2 / n_R)
    // Pure children therefore get the best score (0).
    const double score =
        static_cast<double>(sch) -
        static_cast<double>(left_sq_sum) / left_total -
        static_cast<double>(right_sq_sum) / right_total;

    out_ginis[candidate_idx] = score;
}

__global__ void partition_data_kernel(
    const double* __restrict__ x,
    int m,
    const int* __restrict__ nums,
    int sch,
    double split_val,
    int feature_idx,
    int* left_indices,
    int* right_indices,
    int* left_count,
    int* right_count) {

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= sch) return;

    const int sample_idx = nums[idx];
    const double val = x[sample_idx * m + feature_idx];

    if (val > split_val) {
        const int pos = atomicAdd(right_count, 1);
        right_indices[pos] = sample_idx;
    } else {
        const int pos = atomicAdd(left_count, 1);
        left_indices[pos] = sample_idx;
    }
}

static void check_problem_dimensions(int m, int sch, int noc) {
    if (m <= 0) {
        fprintf(stderr, "m_features must be > 0\n");
        exit(EXIT_FAILURE);
    }
    if (sch <= 0) {
        fprintf(stderr, "subset size must be > 0\n");
        exit(EXIT_FAILURE);
    }
    if (noc <= 0 || noc > MAX_CLASSES) {
        fprintf(stderr,
                "Unsupported number of classes: %d (supported range: 1..%d)\n",
                noc, MAX_CLASSES);
        exit(EXIT_FAILURE);
    }
}

// Return whether all samples in numbers have the same original label.
static int is_list(const int* y, const int* numbers, int sch) {
    if (sch <= 0) return 1;

    const int value = y[numbers[0]];
    for (int i = 1; i < sch; ++i) {
        if (y[numbers[i]] != value) return 0;
    }
    return 1;
}

// Count distinct original labels.
static int getNumOfClass(const int* y, int n) {
    if (n <= 0) return 0;

    std::unordered_map<int, int> labels;
    labels.reserve(static_cast<size_t>(n) * 2);
    for (int i = 0; i < n; ++i) {
        labels.emplace(y[i], 0);
    }
    return static_cast<int>(labels.size());
}

// Convert arbitrary original class labels into dense IDs [0, noc).
// The original y array is left unchanged so leaf predictions remain original labels.
static int* encode_labels(const int* y, int n, int noc) {
    int* encoded = (int*)malloc(static_cast<size_t>(n) * sizeof(int));
    if (!encoded) {
        fprintf(stderr, "Out of host memory while encoding labels\n");
        exit(EXIT_FAILURE);
    }

    std::unordered_map<int, int> label_to_dense;
    label_to_dense.reserve(static_cast<size_t>(noc) * 2);

    int next_id = 0;
    for (int i = 0; i < n; ++i) {
        auto it = label_to_dense.find(y[i]);
        if (it == label_to_dense.end()) {
            if (next_id >= noc) {
                fprintf(stderr, "Internal label encoding error\n");
                free(encoded);
                exit(EXIT_FAILURE);
            }
            it = label_to_dense.emplace(y[i], next_id++).first;
        }
        encoded[i] = it->second;
    }

    return encoded;
}

static int get_value_and_attribute_cuda(
    const double* d_x,
    const double* h_x,
    const int* d_y,
    const int m,
    const int noc,
    const int* d_nums,
    const int* h_nums,
    const int sch,
    double* val,
    int* k,
    double* best_score) {

    check_problem_dimensions(m, sch, noc);

    const int total_candidates = sch * m;
    double* d_ginis = nullptr;

    CUDA_CHECK(cudaMalloc((void**)&d_ginis,
                          static_cast<size_t>(total_candidates) * sizeof(double)));

    const int threads = 256;
    const int blocks = (total_candidates + threads - 1) / threads;

    evaluate_splits_kernel<<<blocks, threads>>>(
        d_x, m, d_y, d_nums, sch, noc, d_ginis);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    double* h_ginis = (double*)malloc(
        static_cast<size_t>(total_candidates) * sizeof(double));
    if (!h_ginis) {
        CUDA_CHECK(cudaFree(d_ginis));
        fprintf(stderr, "Out of host memory for split scores\n");
        exit(EXIT_FAILURE);
    }

    CUDA_CHECK(cudaMemcpy(h_ginis,
                          d_ginis,
                          static_cast<size_t>(total_candidates) * sizeof(double),
                          cudaMemcpyDeviceToHost));

    double min_gini = h_ginis[0];
    int opt_idx = 0;
    for (int i = 1; i < total_candidates; ++i) {
        if (h_ginis[i] < min_gini) {
            min_gini = h_ginis[i];
            opt_idx = i;
        }
    }

    const int best_sample_pos = opt_idx / m;
    *k = opt_idx % m;
    const int best_sample_idx = h_nums[best_sample_pos];

    // h_x is already available on the host, so no extra device-to-host copy is needed.
    *val = h_x[static_cast<size_t>(best_sample_idx) * m + *k];
	*best_score = min_gini;

    free(h_ginis);
    CUDA_CHECK(cudaFree(d_ginis));
	return min_gini < static_cast<double>(sch);
}

static void partition_data_cuda(
    const double* d_x,
    const int m,
    const int* d_nums,
    const int* h_nums,
    const int sch,
    const double split_val,
    const int feature_idx,
    int** left_indices,
    int** right_indices,
    int* left_count,
    int* right_count) {

    int *d_left_indices = nullptr;
    int *d_right_indices = nullptr;
    int *d_left_count = nullptr;
    int *d_right_count = nullptr;

    CUDA_CHECK(cudaMalloc((void**)&d_left_indices,
                          static_cast<size_t>(sch) * sizeof(int)));
    CUDA_CHECK(cudaMalloc((void**)&d_right_indices,
                          static_cast<size_t>(sch) * sizeof(int)));
    CUDA_CHECK(cudaMalloc((void**)&d_left_count, sizeof(int)));
    CUDA_CHECK(cudaMalloc((void**)&d_right_count, sizeof(int)));

    CUDA_CHECK(cudaMemset(d_left_count, 0, sizeof(int)));
    CUDA_CHECK(cudaMemset(d_right_count, 0, sizeof(int)));

    const int threads = 256;
    const int blocks = (sch + threads - 1) / threads;

    partition_data_kernel<<<blocks, threads>>>(
        d_x,
        m,
        d_nums,
        sch,
        split_val,
        feature_idx,
        d_left_indices,
        d_right_indices,
        d_left_count,
        d_right_count);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(left_count,
                          d_left_count,
                          sizeof(int),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(right_count,
                          d_right_count,
                          sizeof(int),
                          cudaMemcpyDeviceToHost));

    *left_indices = nullptr;
    *right_indices = nullptr;

    if (*left_count > 0) {
        *left_indices = (int*)malloc(static_cast<size_t>(*left_count) * sizeof(int));
        if (!*left_indices) {
            fprintf(stderr, "Out of host memory for left partition\n");
            exit(EXIT_FAILURE);
        }
        CUDA_CHECK(cudaMemcpy(*left_indices,
                              d_left_indices,
                              static_cast<size_t>(*left_count) * sizeof(int),
                              cudaMemcpyDeviceToHost));
    }

    if (*right_count > 0) {
        *right_indices = (int*)malloc(static_cast<size_t>(*right_count) * sizeof(int));
        if (!*right_indices) {
            fprintf(stderr, "Out of host memory for right partition\n");
            exit(EXIT_FAILURE);
        }
        CUDA_CHECK(cudaMemcpy(*right_indices,
                              d_right_indices,
                              static_cast<size_t>(*right_count) * sizeof(int),
                              cudaMemcpyDeviceToHost));
    }

    // atomicAdd makes the device output order nondeterministic. Restore the
    // input order so recursive tie-breaking matches the sequential version.
    std::unordered_map<int, int> input_positions;
    input_positions.reserve(static_cast<size_t>(sch) * 2);
    for (int i = 0; i < sch; ++i) {
        input_positions.emplace(h_nums[i], i);
    }
    auto input_order = [&input_positions](int left, int right) {
        return input_positions.at(left) < input_positions.at(right);
    };
    if (*left_count > 1) {
        std::sort(*left_indices, *left_indices + *left_count, input_order);
    }
    if (*right_count > 1) {
        std::sort(*right_indices, *right_indices + *right_count, input_order);
    }

    CUDA_CHECK(cudaFree(d_left_indices));
    CUDA_CHECK(cudaFree(d_right_indices));
    CUDA_CHECK(cudaFree(d_left_count));
    CUDA_CHECK(cudaFree(d_right_count));
}

static int majority_class(
    const int* y,
    const int* numbers,
    int sch) {

    int best_label = y[numbers[0]];
    int best_count = 0;
    for (int i = 0; i < sch; ++i) {
        int count = 0;
        for (int j = 0; j < sch; ++j) {
            if (y[numbers[j]] == y[numbers[i]]) ++count;
        }
        if (count > best_count) {
            best_count = count;
            best_label = y[numbers[i]];
        }
    }
    return best_label;
}

static void create_bin_tree_cuda(
    btree* tree,
    const double* d_x,
    const int* h_y_original,
    const int* d_y_dense,
    const double* h_x,
    const int m,
    const int* numbers,
    const int sch,
    const int noc,
    const int depth) {

    if (!tree || sch <= 0) {
        return;
    }

    tree->data = 0.0;
    tree->left = nullptr;
    tree->right = nullptr;
    tree->num_q = 0;

    if (is_list(h_y_original, numbers, sch) ||
        sch < MIN_SAMPLES_SPLIT ||
        (MAX_DEPTH >= 0 && depth >= MAX_DEPTH)) {
        tree->num_q = majority_class(h_y_original, numbers, sch);
        return;
    }

    check_problem_dimensions(m, sch, noc);

    int* d_nums = nullptr;
    CUDA_CHECK(cudaMalloc((void**)&d_nums,
                          static_cast<size_t>(sch) * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_nums,
                          numbers,
                          static_cast<size_t>(sch) * sizeof(int),
                          cudaMemcpyHostToDevice));

    double val = 0.0;
    double best_score = 0.0;
    int feature_idx = 0;
    int has_split = get_value_and_attribute_cuda(
        d_x,
        h_x,
        d_y_dense,
        m,
        noc,
        d_nums,
        numbers,
        sch,
        &val,
        &feature_idx,
        &best_score);

    double parent_score = static_cast<double>(sch);
    std::unordered_map<int, int> parent_counts;
    for (int i = 0; i < sch; ++i) ++parent_counts[h_y_original[numbers[i]]];
    for (const auto& entry : parent_counts) {
        parent_score -= static_cast<double>(entry.second) * entry.second / sch;
    }
    if (!has_split || best_score >= parent_score - MIN_IMPURITY_DECREASE) {
        tree->num_q = majority_class(h_y_original, numbers, sch);
        CUDA_CHECK(cudaFree(d_nums));
        return;
    }

    tree->data = val;
    tree->num_q = feature_idx;

    int* lefts = nullptr;
    int* rights = nullptr;
    int nol = 0;
    int nor = 0;

    partition_data_cuda(
        d_x,
        m,
        d_nums,
        numbers,
        sch,
        val,
        feature_idx,
        &lefts,
        &rights,
        &nol,
        &nor);

    CUDA_CHECK(cudaFree(d_nums));

    // Defensive handling for a degenerate split.
    if (nol < MIN_SAMPLES_LEAF || nor < MIN_SAMPLES_LEAF) {
        tree->left = nullptr;
        tree->right = nullptr;
        tree->num_q = majority_class(h_y_original, numbers, sch);
        free(lefts);
        free(rights);
        return;
    }

    tree->left = (btree*)malloc(sizeof(btree));
    tree->right = (btree*)malloc(sizeof(btree));
    if (!tree->left || !tree->right) {
        free(tree->left);
        free(tree->right);
        free(lefts);
        free(rights);
        fprintf(stderr, "Out of host memory while growing tree\n");
        exit(EXIT_FAILURE);
    }

    create_bin_tree_cuda(
        tree->left,
        d_x,
        h_y_original,
        d_y_dense,
        h_x,
        m,
        lefts,
        nol,
        noc,
        depth + 1);

    create_bin_tree_cuda(
        tree->right,
        d_x,
        h_y_original,
        d_y_dense,
        h_x,
        m,
        rights,
        nor,
        noc,
        depth + 1);

    free(lefts);
    free(rights);
}

static int get_class(const btree* tree, const double* x) {
    if (!tree) return -1;

    while (tree->left != nullptr && tree->right != nullptr) {
        tree = (x[tree->num_q] > tree->data) ? tree->right : tree->left;
    }
    return tree->num_q;
}

static void get_classes(
    const btree* tree,
    const double* x,
    int* res,
    int n,
    int m) {

    for (int i = 0; i < n; ++i) {
        res[i] = get_class(tree, x + static_cast<size_t>(i) * m);
    }
}

static void free_bin_tree(btree* tree) {
    if (!tree) return;
    free_bin_tree(tree->left);
    free_bin_tree(tree->right);
    free(tree);
}

static void fscanfTrainData(
    double* x,
    int* y,
    int n,
    int m,
    const char* fn) {

    FILE* fl = fopen(fn, "rb");
    if (!fl) {
        fprintf(stderr, "Error opening %s\n", fn);
        exit(EXIT_FAILURE);
    }

    const size_t x_count = static_cast<size_t>(n) * m;
    if (fread(x, sizeof(double), x_count, fl) != x_count) {
        fprintf(stderr, "Error reading training features from %s\n", fn);
        fclose(fl);
        exit(EXIT_FAILURE);
    }

    if (fread(y, sizeof(int), n, fl) != static_cast<size_t>(n)) {
        fprintf(stderr, "Error reading training labels from %s\n", fn);
        fclose(fl);
        exit(EXIT_FAILURE);
    }

    fclose(fl);
}

static void fscanfTestData(double* x, int n_values, const char* fn) {
    FILE* fl = fopen(fn, "rb");
    if (!fl) {
        fprintf(stderr, "Error opening test file %s\n", fn);
        exit(EXIT_FAILURE);
    }

    if (fread(x, sizeof(double), n_values, fl) != static_cast<size_t>(n_values)) {
        fprintf(stderr, "Error reading test data from %s\n", fn);
        fclose(fl);
        exit(EXIT_FAILURE);
    }

    fclose(fl);
}

static void fprintfResult(
    const int* y,
    int n,
    double t1,
    double t2,
    const char* fn) {

    (void)y;
    (void)n;
    (void)t2;

    FILE* fl = fopen(fn, "a");
    if (!fl) {
        fprintf(stderr, "Warning: could not open output file %s\n", fn);
        return;
    }

    fprintf(fl,
            "Result of CART (CUDA Optimized)\n"
            "Tree time: %lf s\n",
            t1);
    for (int i = 0; i < n; ++i) {
        fprintf(fl, "Object[%d]: %d;\n", i + 1, y[i]);
    }
    fclose(fl);
}

int main(int argc, char** argv) {
    if (argc < 7) {
        fprintf(stderr,
                "Usage: %s <n_train> <m_features> <n_test> "
                "<train_file> <test_file> <output_file>\n",
                argv[0]);
        return EXIT_FAILURE;
    }

    const int n = atoi(argv[1]);
    const int m = atoi(argv[2]);
    const int n2 = atoi(argv[3]);

    if (n <= 0 || m <= 0 || n2 < 0) {
        fprintf(stderr, "Invalid dimensions: n=%d m=%d n_test=%d\n", n, m, n2);
        return EXIT_FAILURE;
    }

    int noc = 0;

    double* h_xtrain = (double*)malloc(static_cast<size_t>(n) * m * sizeof(double));
    int* h_y = (int*)malloc(static_cast<size_t>(n) * sizeof(int));
    if (!h_xtrain || !h_y) {
        fprintf(stderr, "Out of host memory for training data\n");
        free(h_xtrain);
        free(h_y);
        return EXIT_FAILURE;
    }

    fscanfTrainData(h_xtrain, h_y, n, m, argv[4]);

    noc = getNumOfClass(h_y, n);
    if (noc <= 0 || noc > MAX_CLASSES) {
        fprintf(stderr,
                "Training data contains %d classes; supported range is 1..%d\n",
                noc,
                MAX_CLASSES);
        free(h_xtrain);
        free(h_y);
        return EXIT_FAILURE;
    }

    // Dense labels are used only inside the GPU Gini calculation.
    int* h_y_dense = encode_labels(h_y, n, noc);

    double* d_xtrain = nullptr;
    int* d_y_dense = nullptr;

    CUDA_CHECK(cudaMalloc((void**)&d_xtrain,
                          static_cast<size_t>(n) * m * sizeof(double)));
    CUDA_CHECK(cudaMalloc((void**)&d_y_dense,
                          static_cast<size_t>(n) * sizeof(int)));

    CUDA_CHECK(cudaMemcpy(d_xtrain,
                          h_xtrain,
                          static_cast<size_t>(n) * m * sizeof(double),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_y_dense,
                          h_y_dense,
                          static_cast<size_t>(n) * sizeof(int),
                          cudaMemcpyHostToDevice));

    double* h_xtest = nullptr;
    if (n2 > 0) {
        h_xtest = (double*)malloc(static_cast<size_t>(n2) * m * sizeof(double));
        if (!h_xtest) {
            fprintf(stderr, "Out of host memory for test data\n");
            CUDA_CHECK(cudaFree(d_xtrain));
            CUDA_CHECK(cudaFree(d_y_dense));
            free(h_y_dense);
            free(h_xtrain);
            free(h_y);
            return EXIT_FAILURE;
        }
        fscanfTestData(h_xtest, n2 * m, argv[5]);
    }

    int* res = nullptr;
    if (n2 > 0) {
        res = (int*)malloc(static_cast<size_t>(n2) * sizeof(int));
        if (!res) {
            fprintf(stderr, "Out of host memory for predictions\n");
            free(h_xtest);
            CUDA_CHECK(cudaFree(d_xtrain));
            CUDA_CHECK(cudaFree(d_y_dense));
            free(h_y_dense);
            free(h_xtrain);
            free(h_y);
            return EXIT_FAILURE;
        }
    }

    const double t_start = static_cast<double>(clock()) / CLOCKS_PER_SEC;

    btree* tree = (btree*)malloc(sizeof(btree));
    if (!tree) {
        fprintf(stderr, "Out of host memory for tree root\n");
        free(res);
        free(h_xtest);
        CUDA_CHECK(cudaFree(d_xtrain));
        CUDA_CHECK(cudaFree(d_y_dense));
        free(h_y_dense);
        free(h_xtrain);
        free(h_y);
        return EXIT_FAILURE;
    }

    int* startNums = (int*)malloc(static_cast<size_t>(n) * sizeof(int));
    if (!startNums) {
        fprintf(stderr, "Out of host memory for sample indices\n");
        free(tree);
        free(res);
        free(h_xtest);
        CUDA_CHECK(cudaFree(d_xtrain));
        CUDA_CHECK(cudaFree(d_y_dense));
        free(h_y_dense);
        free(h_xtrain);
        free(h_y);
        return EXIT_FAILURE;
    }

    for (int i = 0; i < n; ++i) startNums[i] = i;

    create_bin_tree_cuda(
        tree,
        d_xtrain,
        h_y,
        d_y_dense,
        h_xtrain,
        m,
        startNums,
        n,
        noc,
        0);

    CUDA_CHECK(cudaDeviceSynchronize());

    const double t1 = static_cast<double>(clock()) / CLOCKS_PER_SEC - t_start;

    // Build predictions when test data is present.
    if (n2 > 0) {
        get_classes(tree, h_xtest, res, n2, m);
    }

    fprintfResult(res, n2, t1, 0.0, argv[6]);

    printf("\n=== Performance Results ===\n");
    printf("Tree creation time (CUDA Parallelized): %.4f s\n", t1);

    // Uncomment to print predictions if desired.
    // for (int i = 0; i < n2; ++i) printf("%d\n", res[i]);

    free(startNums);
    free(res);
    free(h_xtest);
    free(h_y_dense);
    free(h_xtrain);
    free(h_y);
    free_bin_tree(tree);

    CUDA_CHECK(cudaFree(d_xtrain));
    CUDA_CHECK(cudaFree(d_y_dense));

    return EXIT_SUCCESS;
}