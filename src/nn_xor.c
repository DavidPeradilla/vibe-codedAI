/* Phase 3: a 2-4-1 neural network that learns XOR using backpropagation.
 *
 *   input (2)  ->  hidden layer (4 neurons, sigmoid)  ->  output (1 neuron, sigmoid)
 *
 * We feed all 4 XOR examples at once as a matrix (4 rows, one per example).
 */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "matrix.h"

#define HIDDEN 4

typedef struct {
    Matrix *W1;  /* 2 x HIDDEN */
    Matrix *b1;  /* 1 x HIDDEN */
    Matrix *W2;  /* HIDDEN x 1 */
    Matrix *b2;  /* 1 x 1      */
} Net;

static float sigmoid(float z) { return 1.0f / (1.0f + expf(-z)); }

/* ---------- small helpers (matrix ops the library doesn't have) ---------- */

/* m += bias, where bias is 1 x cols and is added to every row. */
static void add_bias(Matrix *m, const Matrix *bias) {
    for (int i = 0; i < m->rows; i++)
        for (int j = 0; j < m->cols; j++)
            MAT_AT(m, i, j) += bias->data[j];
}

/* Sum each column -> 1 x cols matrix. */
static Matrix *sum_rows(const Matrix *m) {
    Matrix *s = matrix_create(1, m->cols);
    for (int i = 0; i < m->rows; i++)
        for (int j = 0; j < m->cols; j++)
            s->data[j] += MAT_AT(m, i, j);
    return s;
}

/* a * (1 - a) for every element: the derivative of sigmoid, given its output a. */
static float sigmoid_deriv_from_output(float a) { return a * (1.0f - a); }

/* Subtract lr * grad from param, in place. */
static void step(Matrix *param, const Matrix *grad, float lr) {
    for (int i = 0; i < param->rows * param->cols; i++)
        param->data[i] -= lr * grad->data[i];
}

/* ---------- the network ---------- */

static Net net_create(void) {
    Net n;
    n.W1 = matrix_create(2, HIDDEN);
    n.b1 = matrix_create(1, HIDDEN);
    n.W2 = matrix_create(HIDDEN, 1);
    n.b2 = matrix_create(1, 1);
    matrix_randomize(n.W1, -1.0f, 1.0f);
    matrix_randomize(n.W2, -1.0f, 1.0f);
    return n;  /* biases start at 0 */
}

static void net_free(Net *n) {
    matrix_free(n->W1); matrix_free(n->b1);
    matrix_free(n->W2); matrix_free(n->b2);
}

/* Forward pass. Returns hidden activations in *A1 and predictions in *A2. */
static void forward(const Net *n, const Matrix *X, Matrix **A1, Matrix **A2) {
    Matrix *Z1 = matrix_mul(X, n->W1);
    add_bias(Z1, n->b1);
    *A1 = matrix_apply(Z1, sigmoid);

    Matrix *Z2 = matrix_mul(*A1, n->W2);
    add_bias(Z2, n->b2);
    *A2 = matrix_apply(Z2, sigmoid);

    matrix_free(Z1);
    matrix_free(Z2);
}

/* Mean squared error over all examples. */
static float loss_fn(const Matrix *A2, const Matrix *Y) {
    float total = 0.0f;
    for (int i = 0; i < A2->rows; i++) {
        float e = A2->data[i] - Y->data[i];
        total += e * e;
    }
    return total / (float)A2->rows;
}

static float compute_loss(const Net *n, const Matrix *X, const Matrix *Y) {
    Matrix *A1, *A2;
    forward(n, X, &A1, &A2);
    float l = loss_fn(A2, Y);
    matrix_free(A1);
    matrix_free(A2);
    return l;
}

/* Backward pass: computes the gradient of the loss for every parameter. */
static void backward(const Net *n, const Matrix *X, const Matrix *Y,
                     Matrix *A1, Matrix *A2,
                     Matrix **dW1, Matrix **db1, Matrix **dW2, Matrix **db2) {
    int N = X->rows;

    /* Output layer: dZ2 = (2/N) * (A2 - Y) * sigmoid'(Z2) */
    Matrix *dZ2 = matrix_create(N, 1);
    for (int i = 0; i < N; i++) {
        float a = A2->data[i];
        dZ2->data[i] = (2.0f / (float)N) * (a - Y->data[i]) * sigmoid_deriv_from_output(a);
    }

    Matrix *A1T = matrix_transpose(A1);
    *dW2 = matrix_mul(A1T, dZ2);
    *db2 = sum_rows(dZ2);

    /* Push the error back to the hidden layer: dA1 = dZ2 * W2^T */
    Matrix *W2T = matrix_transpose(n->W2);
    Matrix *dA1 = matrix_mul(dZ2, W2T);

    /* dZ1 = dA1 * sigmoid'(Z1) */
    Matrix *dZ1 = matrix_create(dA1->rows, dA1->cols);
    for (int i = 0; i < dZ1->rows * dZ1->cols; i++)
        dZ1->data[i] = dA1->data[i] * sigmoid_deriv_from_output(A1->data[i]);

    Matrix *XT = matrix_transpose(X);
    *dW1 = matrix_mul(XT, dZ1);
    *db1 = sum_rows(dZ1);

    matrix_free(dZ2); matrix_free(A1T); matrix_free(W2T);
    matrix_free(dA1); matrix_free(dZ1); matrix_free(XT);
}

/* Gradient check: compare backprop's answer with a brute-force estimate
   (f(w+e) - f(w-e)) / 2e. If they match, backprop is correct. */
static void gradient_check(Net *n, const Matrix *X, const Matrix *Y) {
    Matrix *A1, *A2, *dW1, *db1, *dW2, *db2;
    forward(n, X, &A1, &A2);
    backward(n, X, Y, A1, A2, &dW1, &db1, &dW2, &db2);

    printf("Gradient check (analytic vs numerical):\n");
    const float eps = 1e-3f;
    for (int k = 0; k < 4; k++) {
        float *w = &n->W1->data[k];
        float orig = *w;
        *w = orig + eps; float lp = compute_loss(n, X, Y);
        *w = orig - eps; float lm = compute_loss(n, X, Y);
        *w = orig;
        float numeric = (lp - lm) / (2.0f * eps);
        printf("  W1[%d]: backprop = %+.6f   numerical = %+.6f\n", k, dW1->data[k], numeric);
    }

    matrix_free(A1); matrix_free(A2);
    matrix_free(dW1); matrix_free(db1); matrix_free(dW2); matrix_free(db2);
    printf("\n");
}

int main(void) {
    srand((unsigned)time(NULL));

    float xv[] = {0, 0,  0, 1,  1, 0,  1, 1};
    float yv[] = {0, 1, 1, 0};
    Matrix *X = matrix_create(4, 2);
    Matrix *Y = matrix_create(4, 1);
    for (int i = 0; i < 8; i++) X->data[i] = xv[i];
    for (int i = 0; i < 4; i++) Y->data[i] = yv[i];

    Net net = net_create();

    gradient_check(&net, X, Y);

    const int epochs = 10000;
    const float lr = 1.0f;

    printf("Training 2-%d-1 network on XOR...\n", HIDDEN);
    for (int epoch = 0; epoch <= epochs; epoch++) {
        Matrix *A1, *A2, *dW1, *db1, *dW2, *db2;
        forward(&net, X, &A1, &A2);

        if (epoch % (epochs / 10) == 0)
            printf("  epoch %5d   loss = %.6f\n", epoch, loss_fn(A2, Y));

        backward(&net, X, Y, A1, A2, &dW1, &db1, &dW2, &db2);
        step(net.W1, dW1, lr); step(net.b1, db1, lr);
        step(net.W2, dW2, lr); step(net.b2, db2, lr);

        matrix_free(A1); matrix_free(A2);
        matrix_free(dW1); matrix_free(db1); matrix_free(dW2); matrix_free(db2);
    }

    Matrix *A1, *A2;
    forward(&net, X, &A1, &A2);
    printf("\nResults:\n");
    int correct = 0;
    for (int i = 0; i < 4; i++) {
        int guess = A2->data[i] >= 0.5f;
        if (guess == (int)Y->data[i]) correct++;
        printf("  %g XOR %g -> %.3f (rounded %d, want %g)\n",
               MAT_AT(X, i, 0), MAT_AT(X, i, 1), A2->data[i], guess, Y->data[i]);
    }
    printf("accuracy: %d/4\n", correct);

    matrix_free(A1); matrix_free(A2);
    matrix_free(X); matrix_free(Y);
    net_free(&net);
    return 0;
}
