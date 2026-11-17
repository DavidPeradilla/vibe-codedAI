#include "matrix.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

Matrix *matrix_create(int rows, int cols) {
    if (rows <= 0 || cols <= 0) return NULL;

    Matrix *m = malloc(sizeof(Matrix));
    if (!m) return NULL;

    m->rows = rows;
    m->cols = cols;
    m->data = calloc((size_t)rows * cols, sizeof(float));
    if (!m->data) {
        free(m);
        return NULL;
    }
    return m;
}

void matrix_free(Matrix *m) {
    if (!m) return;
    free(m->data);
    free(m);
}

Matrix *matrix_copy(const Matrix *m) {
    Matrix *c = matrix_create(m->rows, m->cols);
    if (!c) return NULL;
    memcpy(c->data, m->data, (size_t)m->rows * m->cols * sizeof(float));
    return c;
}

void matrix_fill(Matrix *m, float value) {
    int n = m->rows * m->cols;
    for (int i = 0; i < n; i++) m->data[i] = value;
}

void matrix_randomize(Matrix *m, float min, float max) {
    int n = m->rows * m->cols;
    for (int i = 0; i < n; i++) {
        float r = (float)rand() / (float)RAND_MAX;   /* 0.0 .. 1.0 */
        m->data[i] = min + r * (max - min);
    }
}

/* Shared helper for add / sub / hadamard */
static Matrix *elementwise(const Matrix *a, const Matrix *b, char op) {
    if (a->rows != b->rows || a->cols != b->cols) return NULL;

    Matrix *r = matrix_create(a->rows, a->cols);
    if (!r) return NULL;

    int n = a->rows * a->cols;
    for (int i = 0; i < n; i++) {
        switch (op) {
            case '+': r->data[i] = a->data[i] + b->data[i]; break;
            case '-': r->data[i] = a->data[i] - b->data[i]; break;
            case '*': r->data[i] = a->data[i] * b->data[i]; break;
        }
    }
    return r;
}

Matrix *matrix_add(const Matrix *a, const Matrix *b)      { return elementwise(a, b, '+'); }
Matrix *matrix_sub(const Matrix *a, const Matrix *b)      { return elementwise(a, b, '-'); }
Matrix *matrix_hadamard(const Matrix *a, const Matrix *b) { return elementwise(a, b, '*'); }

Matrix *matrix_mul(const Matrix *a, const Matrix *b) {
    if (a->cols != b->rows) return NULL;   /* (m x n) * (n x p) -> (m x p) */

    Matrix *r = matrix_create(a->rows, b->cols);
    if (!r) return NULL;

    /* i-k-j loop order walks memory sequentially, which is faster than i-j-k. */
    for (int i = 0; i < a->rows; i++) {
        for (int k = 0; k < a->cols; k++) {
            float aik = MAT_AT(a, i, k);
            for (int j = 0; j < b->cols; j++) {
                MAT_AT(r, i, j) += aik * MAT_AT(b, k, j);
            }
        }
    }
    return r;
}

Matrix *matrix_transpose(const Matrix *m) {
    Matrix *t = matrix_create(m->cols, m->rows);
    if (!t) return NULL;

    for (int i = 0; i < m->rows; i++)
        for (int j = 0; j < m->cols; j++)
            MAT_AT(t, j, i) = MAT_AT(m, i, j);
    return t;
}

Matrix *matrix_scale(const Matrix *m, float factor) {
    Matrix *r = matrix_create(m->rows, m->cols);
    if (!r) return NULL;

    int n = m->rows * m->cols;
    for (int i = 0; i < n; i++) r->data[i] = m->data[i] * factor;
    return r;
}

Matrix *matrix_apply(const Matrix *m, float (*fn)(float)) {
    Matrix *r = matrix_create(m->rows, m->cols);
    if (!r) return NULL;

    int n = m->rows * m->cols;
    for (int i = 0; i < n; i++) r->data[i] = fn(m->data[i]);
    return r;
}

void matrix_print(const Matrix *m, const char *name) {
    printf("%s (%d x %d):\n", name, m->rows, m->cols);
    for (int i = 0; i < m->rows; i++) {
        printf("  [ ");
        for (int j = 0; j < m->cols; j++) printf("%8.3f ", MAT_AT(m, i, j));
        printf("]\n");
    }
}
