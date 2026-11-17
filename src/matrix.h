#ifndef MATRIX_H
#define MATRIX_H

/* A matrix stored in row-major order: element (i, j) lives at data[i * cols + j]. */
typedef struct {
    int rows;
    int cols;
    float *data;
} Matrix;

/* Access element (i, j). Works for reading and writing: MAT_AT(m, 1, 2) = 5.0f; */
#define MAT_AT(m, i, j) ((m)->data[(i) * (m)->cols + (j)])

/* Creation and destruction */
Matrix *matrix_create(int rows, int cols);            /* zero-filled; NULL on failure */
void    matrix_free(Matrix *m);                       /* safe to call with NULL */
Matrix *matrix_copy(const Matrix *m);

/* Filling */
void matrix_fill(Matrix *m, float value);
void matrix_randomize(Matrix *m, float min, float max);

/* Math. All of these return a NEW matrix (caller must free it),
   or NULL if the shapes do not match. */
Matrix *matrix_add(const Matrix *a, const Matrix *b);
Matrix *matrix_sub(const Matrix *a, const Matrix *b);
Matrix *matrix_hadamard(const Matrix *a, const Matrix *b);  /* element-wise multiply */
Matrix *matrix_mul(const Matrix *a, const Matrix *b);       /* matrix multiplication */
Matrix *matrix_transpose(const Matrix *m);
Matrix *matrix_scale(const Matrix *m, float factor);
Matrix *matrix_apply(const Matrix *m, float (*fn)(float));  /* apply fn to every element */

/* Debugging */
void matrix_print(const Matrix *m, const char *name);

#endif
