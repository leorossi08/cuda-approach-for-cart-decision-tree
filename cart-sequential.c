/* CART (Classification And Regression Tree) - Sequential Implementation */
/* Combined single-file version */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/* ========== Data Structures ========== */

typedef struct btree {
	double data;
	int num_q;
	struct btree* left;
	struct btree* right;
} btree;

/* ========== Function Declarations ========== */

/* CART algorithm functions */
double calc_gini(const double* const x, const int m, const int* const y, const int noc, const int* const nums, const int sch, const double data, const int k, int* const left, int* const right);
void get_value_and_attribute(const double* const x, const int* const y, const int m, const int noc, const int* const num, const int sch, double* const val, int* const k);
char is_list(const int* const y, const int *numbers, int sch);
void create_bin_tree(btree *tree, const double *x, const int *y, const int m, const int *numbers, const int sch, const int noc);
int get_class(const btree* tree, const double* const x);
void get_classes(const btree* const tree, const double* const x, int* const res, int n, const int m);
void free_bin_tree(btree *tree);

/* Helper functions */
void fscanfTrainData(double *x, int *y, const int n, const int m, const char *fn);
void fscanfTestData(double *x, const int n, const char *fn);
void fscanfIdealSpliting(int *id, const int n, const char *fn);
double calcAccuracy(const int *x, const int *y, const int n);
void fprintfResult(const int *y, const int n, const double t1, const double t2, const char *fn);
void fprintfFullRes(const int *y, const int n, const double a, const double t1, const double t2, const char *fn);
int getNumOfClass(const int* const y, const int n);

/* ========== CART Algorithm Implementation ========== */

double calc_gini(const double* const x, const int m, const int* const y, const int noc, const int* const nums, const int sch, const double data, const int k, int* const left, int* const right) {
	int i, L = 0, R = 0, buf;
	// Could be parallelized
	for (i = 0; i < sch; i++) {
		buf = nums[i];
		if (x[buf * m + k] > data) {
			R++;
			right[y[buf]]++;
		} else {
			L++;
			left[y[buf]]++;
		}
	}
	unsigned int lefts = 0, rights = 0;	
	for (i = 0; i < noc; i++) {
		lefts += left[i] * left[i];
		rights += right[i] * right[i];
	}
	return (L == 0 || R == 0) ? sch : (sch - ((double)lefts / L -  (double)rights / R));
}

void get_value_and_attribute(const double* const x, const int* const y, const int m, const int noc, const int* const num, const int sch, double* const val, int* const k) {
	const size_t size = noc * sizeof(int);
	int *left = (int*)calloc(noc, sizeof(int));
	int *right = (int*)calloc(noc, sizeof(int));
	double opt_data = x[*num * m], cur_gini, min_gini = calc_gini(x, m, y, noc, num, sch, opt_data, 0, left, right);
	int i, j, buf, opt_k = 0;
	// for each sample
	for (j = 0; j < sch; j++) {
		buf = num[j] * m;
		for (i = 0; i < m; i++) {
			// for each feature
			memset(left, 0, size);
			memset(right, 0, size);
			// Calculate Gini impurity
			cur_gini = calc_gini(x, m, y, noc, num, sch, x[buf + i], i, left, right);
			if (cur_gini < min_gini) { // Find Minimum
				min_gini = cur_gini;
				opt_k = i;
				opt_data = x[buf + i];
			}
		}
	}
	free(left);
	free(right);
	*val = opt_data;
	*k = opt_k;
}

char is_list(const int* const y, const int *numbers, int sch) {
	const int value = y[*numbers++];
	while (--sch) {
		if (y[*numbers++] != value) return 0;
	}
	return 1;
}

void create_bin_tree(btree *tree, const double *x, const int *y, const int m, const int *numbers, const int sch, const int noc) {
	if (is_list(y, numbers, sch)) {
		tree->left = NULL;
		tree->right = NULL;
		tree->num_q = y[*numbers];
	} else {
		double val;
		int k;
		get_value_and_attribute(x, y, m, noc, numbers, sch, &val, &k);
		tree->data = val;
		tree->num_q = k;
		int *lefts = (int*)malloc(0 * sizeof(int));
		int *rights = (int*)malloc(0 * sizeof(int));
		int i, nol = 0, nor = 0;
		for (i = 0; i < sch; i++) {
			if (x[numbers[i] * m + k] > val) {
				rights = (int*)realloc(rights, (nor + 1) * sizeof(int));
				rights[nor] = numbers[i];
				nor++;				
			} else {
				lefts = (int*)realloc(lefts, (nol + 1) * sizeof(int));
				lefts[nol] = numbers[i];
				nol++;	
			}
		}
		tree->right = (btree*)malloc(sizeof(btree));
		tree->left = (btree*)malloc(sizeof(btree));
		create_bin_tree(tree->right, x, y, m, rights, nor, noc);
		free(rights);
		create_bin_tree(tree->left, x, y, m, lefts, nol, noc);
		free(lefts);
	}
}

int get_class(const btree* tree, const double* const x) {
	while (tree->left != NULL && tree->right != NULL) tree = (x[tree->num_q] > tree->data) ? tree->right : tree->left;
	return tree->num_q;
}

void get_classes(const btree* const tree, const double* const x, int* const res, int n, const int m) {
	while (n--) res[n] = get_class(tree, x + n * m);
}

void free_bin_tree(btree *tree) {
	if (tree) {
		free_bin_tree(tree->left);
		free_bin_tree(tree->right);
		free(tree);
	}
}

/* ========== Helper Functions Implementation ========== */

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
	fprintf(fl, "Result of CART classification...\n"
		"Creation binary tree time:  %lf s.;\n"
		"Time of receiving classes:  %lf s.;\n"
		"Time of CART:  %lf s.;\n", t1, t2, t1 + t2);
	int i = 0;
	while (i++ < n) fprintf(fl, "Object[%d]: %d;\n", i, *(y++));
	fputc('\n', fl);
	fclose(fl);
}

void fprintfFullRes(const int *y, const int n, const double a, const double t1, const double t2, const char *fn) {
	FILE *fl = fopen(fn, "a");
	if (!fl) {
		printf("Error in opening %s result file\n", fn);
		exit(1);
	}
	fprintf(fl, "Result of CART classification...\n"
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
		exit(1);
	}
	const int n = atoi(argv[1]), m = atoi(argv[2]), n2 = atoi(argv[3]);
	int i, noc;
	double *xtrain = (double*)malloc(n * m * sizeof(double));
	int *y = (int *)malloc(n * sizeof(int));
	fscanfTrainData(xtrain, y, n, m, argv[4]);
	double *xtest = (double*)malloc(n2 * m * sizeof(double));
	fscanfTestData(xtest, n2 * m, argv[5]);
	int *res = (int*)malloc(n2 * sizeof(int));
	double t1, t2;
	t1 = clock();
	btree *tree = (btree*)malloc(sizeof(btree));
	noc = getNumOfClass(y, n);
	int *startNums = (int*)malloc(n * sizeof(int));
	for (i = 0; i < n; i++) {
		startNums[i] = i;
	}
	create_bin_tree(tree, xtrain, y, m, startNums, n, noc);
	t1 = clock() - t1;
	t1 /= CLOCKS_PER_SEC;
	t2 = clock();
	get_classes(tree, xtest, res, n2, m);
	t2 = clock() - t2;
	t2 /= CLOCKS_PER_SEC;
	if (argc > 7) {
		int *id = (int*)malloc(n2 * sizeof(int));
		fscanfIdealSpliting(id, n2, argv[7]);
		double a = calcAccuracy(res, id, n2);
		fprintfFullRes(res, n2, a, t1, t2, argv[6]);
		free(id);
		printf("Accuracy of classification by CART  = %lf;\n", a);
	} else {
		fprintfResult(res, n2, t1, t2, argv[6]);
	}
	printf("Creation binary tree time:  %lf s.;\n", t1);
	printf("Time of receiving classes:  %lf s.;\n", t2);
	printf("Time of CART:  %lf s.;\n", t1 + t2);
	printf("The work of the program is completed\n");
	free(startNums);
	free(res);
	free(xtest);
	free(xtrain);
	free(y);
	free_bin_tree(tree);
	return 0;
}
