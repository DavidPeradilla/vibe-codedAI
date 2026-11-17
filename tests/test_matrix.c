#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "matrix.h"

static int tests_run = 0, tests_failed = 0;

#define CHECK(cond, msg)                                         \
    do {                                                         \
        tests_run++;                                             \
        if (!(cond)) {                                           \
            tests_failed++;                                      \
            printf("  FAIL: %s (line %d)\n", msg, __LINE__);     \
        } else {                                                 \
            printf("  ok:   %s\n", msg);                         \
        }                                                        \
    } while (0)

/* Build a matrix from a plain array, to keep tests short. */
static Matrix *from_array(int rows, int cols, const float *vals) {
    Matrix *m = matrix_create(rows, cols);
    for (int i = 0; i < rows * cols; i++) m->data[i] = vals[i];
    return m;
}

static int equals(const Matrix *m, const float *vals) {
    for (int i = 0; i < m->rows * m->cols; i++)
        if (fabsf(m->data[i] - vals[i]) > 1e-5f) return 0;
    return 1;
}

static float square(float x) { return x * x; }

int main(void) {
    printf("Testing matrix library\n");

    /* create / access */
    Matrix *z = matrix_create(2, 3);
    CHECK(z && z->rows == 2 && z->cols == 3, "create sets shape");
    CHECK(MAT_AT(z, 1, 2) == 0.0f, "create zero-fills");
    MAT_AT(z, 1, 2) = 7.0f;
    CHECK(z->data[1 * 3 + 2] == 7.0f, "MAT_AT writes row-major");
    CHECK(matrix_create(0, 5) == NULL, "create rejects zero size");
    matrix_free(z);
    matrix_free(NULL);  /* must not crash */

    /* add / sub / hadamard */
    float av[] = {1, 2, 3, 4}, bv[] = {5, 6, 7, 8};
    Matrix *a = from_array(2, 2, av), *b = from_array(2, 2, bv);

    float add_exp[] = {6, 8, 10, 12};
    Matrix *r = matrix_add(a, b);
    CHECK(equals(r, add_exp), "add");
    matrix_free(r);

    float sub_exp[] = {-4, -4, -4, -4};
    r = matrix_sub(a, b);
    CHECK(equals(r, sub_exp), "sub");
    matrix_free(r);

    float had_exp[] = {5, 12, 21, 32};
    r = matrix_hadamard(a, b);
    CHECK(equals(r, had_exp), "hadamard");
    matrix_free(r);

    Matrix *c3 = matrix_create(2, 3);
    CHECK(matrix_add(a, c3) == NULL, "add rejects shape mismatch");
    matrix_free(c3);

    /* matmul: (2x3) * (3x2) */
    float m1v[] = {1, 2, 3, 4, 5, 6};
    float m2v[] = {7, 8, 9, 10, 11, 12};
    Matrix *m1 = from_array(2, 3, m1v), *m2 = from_array(3, 2, m2v);
    float mul_exp[] = {58, 64, 139, 154};   /* worked out by hand */
    r = matrix_mul(m1, m2);
    CHECK(r && r->rows == 2 && r->cols == 2 && equals(r, mul_exp), "matmul 2x3 * 3x2");
    matrix_free(r);
    CHECK(matrix_mul(m1, m1) == NULL, "matmul rejects bad shapes");

    /* identity */
    float idv[] = {1, 0, 0, 1};
    Matrix *id = from_array(2, 2, idv);
    r = matrix_mul(a, id);
    CHECK(equals(r, av), "A * I == A");
    matrix_free(r);

    /* transpose */
    float t_exp[] = {1, 4, 2, 5, 3, 6};
    r = matrix_transpose(m1);
    CHECK(r->rows == 3 && r->cols == 2 && equals(r, t_exp), "transpose");
    Matrix *rr = matrix_transpose(r);
    CHECK(equals(rr, m1v), "transpose twice == original");
    matrix_free(r); matrix_free(rr);

    /* scale / apply / copy */
    float sc_exp[] = {2, 4, 6, 8};
    r = matrix_scale(a, 2.0f);
    CHECK(equals(r, sc_exp), "scale");
    matrix_free(r);

    float sq_exp[] = {1, 4, 9, 16};
    r = matrix_apply(a, square);
    CHECK(equals(r, sq_exp), "apply(square)");
    matrix_free(r);

    r = matrix_copy(a);
    r->data[0] = 99;
    CHECK(a->data[0] == 1.0f, "copy is independent");
    matrix_free(r);

    /* randomize range */
    Matrix *rnd = matrix_create(10, 10);
    matrix_randomize(rnd, -1.0f, 1.0f);
    int in_range = 1;
    for (int i = 0; i < 100; i++)
        if (rnd->data[i] < -1.0f || rnd->data[i] > 1.0f) in_range = 0;
    CHECK(in_range, "randomize stays within [min, max]");
    matrix_free(rnd);

    matrix_free(a); matrix_free(b); matrix_free(m1); matrix_free(m2); matrix_free(id);

    printf("\n%d tests, %d failed\n", tests_run, tests_failed);
    return tests_failed ? 1 : 0;
}
