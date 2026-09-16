#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/* ========== Macro para verificação de erros CUDA ========== */
#define CUDA_CHECK(call) \
do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error in %s:%d: %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
} while(0)

/* ========== Data Structures ========== */
typedef struct btree {
    double data;
    int num_q;
    struct btree* left;
    struct btree* right;
} btree;

/* ========== Individual CUDA Kernels ========== */

// 1. Kernel to calculate counts for left/right splits
__global__ void calc_gini_counts_kernel(const double* x, const int m, 
                                         const int* y, const int* nums, 
                                         const int sch, const double split_val, 
                                         const int feature_idx, const int noc,
                                         int* left_counts, int* right_counts,
                                         int* L, int* R) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Shared memory for block-level reduction
    extern __shared__ int shared_mem[];
    int* s_left = shared_mem;
    int* s_right = &shared_mem[noc];
    int* s_L = &shared_mem[2 * noc];
    int* s_R = &shared_mem[2 * noc + 1];
    
    // Initialize shared memory
    if (threadIdx.x < noc) {
        s_left[threadIdx.x] = 0;
        s_right[threadIdx.x] = 0;
    }
    if (threadIdx.x == 0) {
        s_L[0] = 0;
        s_R[0] = 0;
    }
    __syncthreads();
    
    // Each thread processes one sample
    if (idx < sch) {
        int sample_idx = nums[idx];
        double feature_val = x[sample_idx * m + feature_idx];
        int class_label = y[sample_idx];
        
        if (feature_val > split_val) {
            atomicAdd(&s_right[class_label], 1);
            atomicAdd(&s_R[0], 1);
        } else {
            atomicAdd(&s_left[class_label], 1);
            atomicAdd(&s_L[0], 1);
        }
    }
    __syncthreads();
    
    // Write block results to global memory
    if (threadIdx.x < noc) {
        atomicAdd(&left_counts[threadIdx.x], s_left[threadIdx.x]);
        atomicAdd(&right_counts[threadIdx.x], s_right[threadIdx.x]);
    }
    if (threadIdx.x == 0) {
        atomicAdd(L, s_L[0]);
        atomicAdd(R, s_R[0]);
    }
}

// 4. Kernel to partition data into left/right based on split
__global__ void partition_data_kernel(const double* x, const int m,
                                      const int* nums, const int sch,
                                      const double split_val, 
                                      const int feature_idx,
                                      int* left_indices, int* right_indices,
                                      int* left_count, int* right_count) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < sch) {
        int sample_idx = nums[idx];
        double val = x[sample_idx * m + feature_idx];
        
        if (val > split_val) {
            int pos = atomicAdd(right_count, 1);
            right_indices[pos] = sample_idx;
        } else {
            int pos = atomicAdd(left_count, 1);
            left_indices[pos] = sample_idx;
        }
    }
}

/* ========== CUDA Helper Functions ========== */

// Host function to calculate Gini using CUDA
double calc_gini_cuda(const double* d_x, const int m, const int* d_y,
                      const int noc, const int* d_nums, const int sch,
                      const double split_val, const int feature_idx) {
    // Allocate device memory
    int *d_left, *d_right, *d_L, *d_R;
    CUDA_CHECK(cudaMalloc(&d_left, noc * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_right, noc * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_L, sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_R, sizeof(int)));
    
    CUDA_CHECK(cudaMemset(d_left, 0, noc * sizeof(int)));
    CUDA_CHECK(cudaMemset(d_right, 0, noc * sizeof(int)));
    CUDA_CHECK(cudaMemset(d_L, 0, sizeof(int)));
    CUDA_CHECK(cudaMemset(d_R, 0, sizeof(int)));
    
    // Launch kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (sch + threadsPerBlock - 1) / threadsPerBlock;
    int sharedMemSize = (2 * noc + 2) * sizeof(int);
    
    calc_gini_counts_kernel<<<blocksPerGrid, threadsPerBlock, sharedMemSize>>>(
        d_x, m, d_y, d_nums, sch, split_val, feature_idx, noc,
        d_left, d_right, d_L, d_R
    );
    
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Copy results back
    int *left = (int*)malloc(noc * sizeof(int));
    int *right = (int*)malloc(noc * sizeof(int));
    int L, R;

    CUDA_CHECK(cudaMemcpy(left, d_left, noc * sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(right, d_right, noc * sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&L, d_L, sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&R, d_R, sizeof(int), cudaMemcpyDeviceToHost));
    
    // Calculate Gini
    unsigned int lefts = 0, rights = 0;
    for (int i = 0; i < noc; i++) {
        lefts += left[i] * left[i];
        rights += right[i] * right[i];
    }
    
    double gini = (L == 0 || R == 0) ? sch : 
                  (sch - ((double)lefts / L - (double)rights / R));
    
    // Cleanup
    free(left);
    free(right);
    CUDA_CHECK(cudaFree(d_left));
    CUDA_CHECK(cudaFree(d_right));
    CUDA_CHECK(cudaFree(d_L));
    CUDA_CHECK(cudaFree(d_R));

    return gini;
}

// Get best split using CUDA-accelerated Gini calculation
void get_value_and_attribute_cuda(const double* d_x, const int* d_y, 
                                   const int m, const int noc, 
                                   const int* d_nums, const int sch, 
                                   double* val, int* k) {
    double min_gini = 1e9;
    int opt_k = 0;
    double opt_data = 0.0;
    
    // Copy nums to host to access sample indices
    int* h_nums = (int*)malloc(sch * sizeof(int));
    CUDA_CHECK(cudaMemcpy(h_nums, d_nums, sch * sizeof(int), cudaMemcpyDeviceToHost));
    
    // For each sample
    for (int j = 0; j < sch; j++) {
        int sample_idx = h_nums[j];
        
        // For each feature
        for (int i = 0; i < m; i++) {
            // Get split value from device (simplified - in production use batched access)
            double split_val;
            CUDA_CHECK(cudaMemcpy(&split_val, &d_x[sample_idx * m + i], 
                                 sizeof(double), cudaMemcpyDeviceToHost));
            
            // Calculate Gini using CUDA
            double cur_gini = calc_gini_cuda(d_x, m, d_y, noc, d_nums, 
                                            sch, split_val, i);
            
            if (cur_gini < min_gini) {
                min_gini = cur_gini;
                opt_k = i;
                opt_data = split_val;
            }
        }
    }
    
    free(h_nums);
    *val = opt_data;
    *k = opt_k;
}

// Partition data using CUDA
void partition_data_cuda(const double* d_x, const int m, const int* d_nums, 
                        const int sch, const double split_val, const int feature_idx,
                        int** left_indices, int** right_indices, 
                        int* left_count, int* right_count) {
    // Allocate device memory
    int *d_left_indices, *d_right_indices, *d_left_count, *d_right_count;
    CUDA_CHECK(cudaMalloc(&d_left_indices, sch * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_right_indices, sch * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_left_count, sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_right_count, sizeof(int)));
    
    CUDA_CHECK(cudaMemset(d_left_count, 0, sizeof(int)));
    CUDA_CHECK(cudaMemset(d_right_count, 0, sizeof(int)));
    
    // Launch kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (sch + threadsPerBlock - 1) / threadsPerBlock;
    
    partition_data_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        d_x, m, d_nums, sch, split_val, feature_idx,
        d_left_indices, d_right_indices, d_left_count, d_right_count
    );
    
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Copy counts back
    CUDA_CHECK(cudaMemcpy(left_count, d_left_count, sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(right_count, d_right_count, sizeof(int), cudaMemcpyDeviceToHost));
    
    // Allocate host memory and copy indices
    *left_indices = (int*)malloc(*left_count * sizeof(int));
    *right_indices = (int*)malloc(*right_count * sizeof(int));
    
    CUDA_CHECK(cudaMemcpy(*left_indices, d_left_indices, 
                         *left_count * sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(*right_indices, d_right_indices, 
                         *right_count * sizeof(int), cudaMemcpyDeviceToHost));
    
    // Cleanup
    CUDA_CHECK(cudaFree(d_left_indices));
    CUDA_CHECK(cudaFree(d_right_indices));
    CUDA_CHECK(cudaFree(d_left_count));
    CUDA_CHECK(cudaFree(d_right_count));
}

/* ========== CPU Functions (kept from sequential) ========== */

char is_list(const int* const y, const int *numbers, int sch) {
    const int value = y[*numbers++];
    while (--sch) {
        if (y[*numbers++] != value) return 0;
    }
    return 1;
}

void create_bin_tree_cuda(btree *tree, const double *d_x, const int *h_y, 
                          const int m, const int *numbers, const int sch, 
                          const int noc, const double* h_x, const int* d_y) {
    if (is_list(h_y, numbers, sch)) {
        tree->left = NULL;
        tree->right = NULL;
        tree->num_q = h_y[*numbers];
    } else {
        double val;
        int k;
        
        // Upload current sample indices to device
        int* d_nums;
        CUDA_CHECK(cudaMalloc(&d_nums, sch * sizeof(int)));
        CUDA_CHECK(cudaMemcpy(d_nums, numbers, sch * sizeof(int), cudaMemcpyHostToDevice));
        
        // Get best split using CUDA
        get_value_and_attribute_cuda(d_x, d_y, m, noc, d_nums, sch, &val, &k);
        
        tree->data = val;
        tree->num_q = k;
        
        // Partition data using CUDA
        int *lefts, *rights;
        int nol, nor;
        partition_data_cuda(d_x, m, d_nums, sch, val, k, &lefts, &rights, &nol, &nor);
        
        CUDA_CHECK(cudaFree(d_nums));
        
        // Recursively build subtrees
        tree->right = (btree*)malloc(sizeof(btree));
        tree->left = (btree*)malloc(sizeof(btree));
        create_bin_tree_cuda(tree->right, d_x, h_y, m, rights, nor, noc, h_x, d_y);
        free(rights);
        create_bin_tree_cuda(tree->left, d_x, h_y, m, lefts, nol, noc, h_x, d_y);
        free(lefts);
    }
}

int get_class(const btree* tree, const double* const x) {
    while (tree->left != NULL && tree->right != NULL) {
        tree = (x[tree->num_q] > tree->data) ? tree->right : tree->left;
    }
    return tree->num_q;
}

void get_classes(const btree* const tree, const double* const x, 
                 int* const res, int n, const int m) {
    while (n--) res[n] = get_class(tree, x + n * m);
}

void free_bin_tree(btree *tree) {
    if (tree) {
        free_bin_tree(tree->left);
        free_bin_tree(tree->right);
        free(tree);
    }
}

/* ========== Helper Functions ========== */

void fscanfTrainData(double *x, int *y, const int n, const int m, const char *fn) {
    FILE *fl = fopen(fn, "rb"); // "rb" = Read Binary
    if (!fl) {
        printf("Error in opening %s file...\n", fn);
        exit(1);
    }
    
    // Lê TODAS as features de uma vez só (n amostras * m features)
    if (fread(x, sizeof(double), (size_t)n * m, fl) == 0) {}
    
    // Lê TODOS os targets/labels de uma vez só (n amostras)
    if (fread(y, sizeof(int), n, fl) == 0) {}
    
    fclose(fl);
}

void fscanfTestData(double *x, const int n, const char *fn) {
    FILE *fl = fopen(fn, "rb"); // "rb" = Read Binary
    if (!fl) {
        printf("Error in opening %s file...\n", fn);
        exit(1);
    }
    
    // Na sua main(), 'n' já é passado como (n2 * m), então basta ler
    if (fread(x, sizeof(double), n, fl) == 0) {}
    
    fclose(fl);
}



void fscanfIdealSpliting(int *id, const int n, const char *fn) {
    FILE *fl = fopen(fn, "r");
    if (!fl) {
        printf("Error in opening %s file...\n", fn);
        exit(1);
    }
    int i;
    for (i = 0; i < n && !feof(fl); i++) {
        if (fscanf(fl, "%d", id + i) == 0) {}
    }
    fclose(fl);
}

double calcAccuracy(const int *x, const int *y, const int n) {
    int i = 0, j = 0;
    while (i++ < n) if (*(x++) == *(y++)) j++;
    return (double)j / (double)n;
}

void fprintfResult(const int *y, const int n, const double t1, const double t2, const char *fn) {
    FILE *fl = fopen(fn, "a");
    if (!fl) {
        printf("Error in opening %s result file\n", fn);
        exit(1);
    }
    fprintf(fl, "Result of CART classification (CUDA-accelerated)...\n"
            "Creation binary tree time:  %lf s.;\n"
            "Time of receiving classes:  %lf s.;\n"
            "Time of CART:  %lf s.;\n", t1, t2, t1 + t2);
    int i = 0;
    while (i++ < n) fprintf(fl, "Object[%d]: %d;\n", i, *(y++));
    fputc('\n', fl);
    fclose(fl);
}

void fprintfFullRes(const int *y, const int n, const double a, const double t1, 
                    const double t2, const char *fn) {
    FILE *fl = fopen(fn, "a");
    if (!fl) {
        printf("Error in opening %s result file\n", fn);
        exit(1);
    }
    fprintf(fl, "Result of CART classification (CUDA-accelerated)...\n"
            "Accuracy of classification = %lf;\n"
            "Creation binary tree time:  %lf s.;\n"
            "Time of receiving classes:  %lf s.;\n"
            "Time of CART:  %lf s.;\n", a, t1, t2, t1 + t2);
    int i = 0;
    while (i++ < n) fprintf(fl, "Object[%d]: %d;\n", i, *(y++));
    fputc('\n', fl);
    fclose(fl);
}

int getNumOfClass(const int* const y, const int n) {
    int i, j, cur;
    char *v = (char*)malloc(n * sizeof(char));
    memset(v, 0, n * sizeof(char));
    for (i = 0; i < n; i++) {
        while (i < n && v[i]) i++;
        cur = y[i];
        for (j = i + 1; j < n; j++) {
            if (y[j] == cur) v[j] = 1;
        }
    }
    i = cur = 0;
    while (i < n) {
        if (v[i] == 0) cur++;
        i++;
    }
    free(v);
    return cur;
}

/* ========== Main Function ========== */

int main(int argc, char **argv) {
    if (argc < 7) {
        printf("Not enough parameters...\n");
        printf("Usage: %s <n_train> <m_features> <n_test> <train_file> <test_file> <output_file> [ideal_file]\n", argv[0]);
        exit(1);
    }
    
    // Display GPU information
    int deviceCount;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    if (deviceCount == 0) {
        printf("No CUDA devices found!\n");
        exit(1);
    }
    
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("\n=== GPU Information ===\n");
    printf("Device: %s\n", prop.name);
    printf("Compute Capability: %d.%d\n", prop.major, prop.minor);
    printf("Multiprocessors: %d\n", prop.multiProcessorCount);
    printf("Global Memory: %.2f GB\n\n", prop.totalGlobalMem / 1e9);
    
    const int n = atoi(argv[1]), m = atoi(argv[2]), n2 = atoi(argv[3]);
    int i, noc;
    
    // Allocate and load training data
    printf("Loading training data...\n");
    double *h_xtrain = (double*)malloc(n * m * sizeof(double));
    int *h_y = (int*)malloc(n * sizeof(int));
    fscanfTrainData(h_xtrain, h_y, n, m, argv[4]);
    
    // Upload training data to GPU
    printf("Uploading data to GPU...\n");
    double *d_xtrain;
    int *d_y;
    CUDA_CHECK(cudaMalloc(&d_xtrain, n * m * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_y, n * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_xtrain, h_xtrain, n * m * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_y, h_y, n * sizeof(int), cudaMemcpyHostToDevice));
    
    // Load test data
    printf("Loading test data...\n");
    double *h_xtest = (double*)malloc(n2 * m * sizeof(double));
    fscanfTestData(h_xtest, n2 * m, argv[5]);
    
    int *res = (int*)malloc(n2 * sizeof(int));
    double t1, t2;
    
    // Build decision tree using CUDA-accelerated functions
    printf("Building decision tree with CUDA acceleration...\n");
    t1 = clock();
    btree *tree = (btree*)malloc(sizeof(btree));
    noc = getNumOfClass(h_y, n);
    int *startNums = (int*)malloc(n * sizeof(int));
    for (i = 0; i < n; i++) {
        startNums[i] = i;
    }
    create_bin_tree_cuda(tree, d_xtrain, h_y, m, startNums, n, noc, h_xtrain, d_y);
    t1 = clock() - t1;
    t1 /= CLOCKS_PER_SEC;
    
    // Classify test data
    printf("Classifying test data...\n");
    t2 = clock();
    get_classes(tree, h_xtest, res, n2, m);
    t2 = clock() - t2;
    t2 /= CLOCKS_PER_SEC;
    
    // Calculate and display results
    if (argc > 7) {
        int *id = (int*)malloc(n2 * sizeof(int));
        fscanfIdealSpliting(id, n2, argv[7]);
        double a = calcAccuracy(res, id, n2);
        fprintfFullRes(res, n2, a, t1, t2, argv[6]);
        free(id);
        printf("\nAccuracy of classification by CART = %.4lf (%.2f%%)\n", a, a * 100);
    } else {
        fprintfResult(res, n2, t1, t2, argv[6]);
    }
    
    printf("\n=== Performance Results ===\n");
    printf("Tree creation time (CUDA): %.4lf s\n", t1);
    printf("Classification time: %.4lf s\n", t2);
    printf("Total CART time: %.4lf s\n", t1 + t2);
    printf("\nThe work of the program is completed\n");
    
    // Cleanup
    free(startNums);
    free(res);
    free(h_xtest);
    free(h_xtrain);
    free(h_y);
    free_bin_tree(tree);
    
    CUDA_CHECK(cudaFree(d_xtrain));
    CUDA_CHECK(cudaFree(d_y));
    CUDA_CHECK(cudaDeviceReset());
    
    return 0;
}
