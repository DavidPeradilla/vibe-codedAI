/* Phase 4: train a 784-128-10 network on MNIST handwritten digits.
 *
 *   784 pixels -> 128 hidden (sigmoid) -> 10 outputs (softmax)
 *
 * Training uses mini-batches: we update the weights after every 64 images
 * instead of after the whole dataset (stochastic gradient descent).
 */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "matrix.h"
#include "mnist.h"

#define HIDDEN     128
#define CLASSES    10
#define BATCH      64
#define EPOCHS     5
#define LEARN_RATE 0.5f

typedef struct {
    Matrix *W1, *b1, *W2, *b2;
} Net;

static float sigmoid(float z) { return 1.0f / (1.0f + expf(-z)); }

static void add_bias(Matrix *m, const Matrix *bias) {
    for (int i = 0; i < m->rows; i++)
        for (int j = 0; j < m->cols; j++)
            MAT_AT(m, i, j) += bias->data[j];
}

static Matrix *sum_rows(const Matrix *m) {
    Matrix *s = matrix_create(1, m->cols);
    for (int i = 0; i < m->rows; i++)
        for (int j = 0; j < m->cols; j++)
            s->data[j] += MAT_AT(m, i, j);
    return s;
}

static void step(Matrix *param, const Matrix *grad, float lr) {
    for (int i = 0; i < param->rows * param->cols; i++)
        param->data[i] -= lr * grad->data[i];
}

/* Softmax turns 10 raw scores per row into 10 probabilities that sum to 1.
   We subtract the row max first so exp() never overflows. */
static void softmax_rows(Matrix *m) {
    for (int i = 0; i < m->rows; i++) {
        float max = MAT_AT(m, i, 0);
        for (int j = 1; j < m->cols; j++)
            if (MAT_AT(m, i, j) > max) max = MAT_AT(m, i, j);

        float sum = 0.0f;
        for (int j = 0; j < m->cols; j++) {
            MAT_AT(m, i, j) = expf(MAT_AT(m, i, j) - max);
            sum += MAT_AT(m, i, j);
        }
        for (int j = 0; j < m->cols; j++) MAT_AT(m, i, j) /= sum;
    }
}

static Net net_create(void) {
    Net n;
    n.W1 = matrix_create(MNIST_PIXELS, HIDDEN);
    n.b1 = matrix_create(1, HIDDEN);
    n.W2 = matrix_create(HIDDEN, CLASSES);
    n.b2 = matrix_create(1, CLASSES);

    /* Xavier initialization: smaller starting weights when a layer has more inputs. */
    float lim1 = sqrtf(6.0f / (MNIST_PIXELS + HIDDEN));
    float lim2 = sqrtf(6.0f / (HIDDEN + CLASSES));
    matrix_randomize(n.W1, -lim1, lim1);
    matrix_randomize(n.W2, -lim2, lim2);
    return n;
}

static void net_free(Net *n) {
    matrix_free(n->W1); matrix_free(n->b1);
    matrix_free(n->W2); matrix_free(n->b2);
}

/* Forward pass. *A1 = hidden activations, *P = class probabilities. */
static void forward(const Net *n, const Matrix *X, Matrix **A1, Matrix **P) {
    Matrix *Z1 = matrix_mul(X, n->W1);
    add_bias(Z1, n->b1);
    *A1 = matrix_apply(Z1, sigmoid);

    *P = matrix_mul(*A1, n->W2);
    add_bias(*P, n->b2);
    softmax_rows(*P);

    matrix_free(Z1);
}

/* Copy the images/labels at the given indices into a batch matrix X and one-hot matrix Y. */
static void fill_batch(Matrix *X, Matrix *Y, const float *images,
                       const unsigned char *labels, const int *idx, int n) {
    matrix_fill(Y, 0.0f);
    for (int b = 0; b < n; b++) {
        memcpy(&X->data[b * MNIST_PIXELS], &images[(size_t)idx[b] * MNIST_PIXELS],
               MNIST_PIXELS * sizeof(float));
        MAT_AT(Y, b, labels[idx[b]]) = 1.0f;
    }
}

/* One training step on one mini-batch. Returns the batch's cross-entropy loss. */
static float train_batch(Net *n, const Matrix *X, const Matrix *Y) {
    int B = X->rows;
    Matrix *A1, *P;
    forward(n, X, &A1, &P);

    /* Cross-entropy loss: -log(probability given to the correct digit) */
    float loss = 0.0f;
    for (int i = 0; i < B; i++)
        for (int j = 0; j < CLASSES; j++)
            if (MAT_AT(Y, i, j) > 0.5f) loss -= logf(MAT_AT(P, i, j) + 1e-9f);
    loss /= (float)B;

    /* For softmax + cross-entropy the gradient simplifies beautifully:
       dZ2 = (P - Y) / B */
    Matrix *dZ2 = matrix_sub(P, Y);
    for (int i = 0; i < dZ2->rows * dZ2->cols; i++) dZ2->data[i] /= (float)B;

    Matrix *A1T = matrix_transpose(A1);
    Matrix *dW2 = matrix_mul(A1T, dZ2);
    Matrix *db2 = sum_rows(dZ2);

    Matrix *W2T = matrix_transpose(n->W2);
    Matrix *dA1 = matrix_mul(dZ2, W2T);
    for (int i = 0; i < dA1->rows * dA1->cols; i++) {
        float a = A1->data[i];
        dA1->data[i] *= a * (1.0f - a);              /* dZ1 (in place) */
    }
    Matrix *XT = matrix_transpose(X);
    Matrix *dW1 = matrix_mul(XT, dA1);
    Matrix *db1 = sum_rows(dA1);

    step(n->W1, dW1, LEARN_RATE); step(n->b1, db1, LEARN_RATE);
    step(n->W2, dW2, LEARN_RATE); step(n->b2, db2, LEARN_RATE);

    matrix_free(A1); matrix_free(P); matrix_free(dZ2); matrix_free(A1T);
    matrix_free(dW2); matrix_free(db2); matrix_free(W2T); matrix_free(dA1);
    matrix_free(XT); matrix_free(dW1); matrix_free(db1);
    return loss;
}

/* Fraction of images whose highest-probability class equals the true label. */
static float accuracy(const Net *n, const float *images, const unsigned char *labels, int count) {
    const int CHUNK = 500;
    int correct = 0;
    Matrix *X = matrix_create(CHUNK, MNIST_PIXELS);

    for (int start = 0; start < count; start += CHUNK) {
        int m = (count - start < CHUNK) ? count - start : CHUNK;
        memcpy(X->data, &images[(size_t)start * MNIST_PIXELS],
               (size_t)m * MNIST_PIXELS * sizeof(float));
        X->rows = m;                                  /* use only the first m rows */

        Matrix *A1, *P;
        forward(n, X, &A1, &P);
        for (int i = 0; i < m; i++) {
            int best = 0;
            for (int j = 1; j < CLASSES; j++)
                if (MAT_AT(P, i, j) > MAT_AT(P, i, best)) best = j;
            if (best == labels[start + i]) correct++;
        }
        matrix_free(A1); matrix_free(P);
    }
    X->rows = CHUNK;
    matrix_free(X);
    return (float)correct / (float)count;
}

static void shuffle(int *a, int n) {
    for (int i = n - 1; i > 0; i--) {
        int j = rand() % (i + 1);
        int t = a[i]; a[i] = a[j]; a[j] = t;
    }
}

int main(int argc, char **argv) {
    const char *dir = (argc > 1) ? argv[1] : "data";
    char path[512];
    srand((unsigned)time(NULL));

    int n_train, n_train_l, n_test, n_test_l;
    snprintf(path, sizeof path, "%s/train-images-idx3-ubyte", dir);
    float *train_x = mnist_load_images(path, &n_train);
    snprintf(path, sizeof path, "%s/train-labels-idx1-ubyte", dir);
    unsigned char *train_y = mnist_load_labels(path, &n_train_l);
    snprintf(path, sizeof path, "%s/t10k-images-idx3-ubyte", dir);
    float *test_x = mnist_load_images(path, &n_test);
    snprintf(path, sizeof path, "%s/t10k-labels-idx1-ubyte", dir);
    unsigned char *test_y = mnist_load_labels(path, &n_test_l);

    if (!train_x || !train_y || !test_x || !test_y || n_train != n_train_l || n_test != n_test_l) {
        fprintf(stderr, "\nCould not load MNIST from '%s'. Run the download script first.\n", dir);
        return 1;
    }
    printf("Loaded %d training and %d test images.\n\n", n_train, n_test);

    Net net = net_create();
    int *order = malloc((size_t)n_train * sizeof(int));
    for (int i = 0; i < n_train; i++) order[i] = i;

    Matrix *X = matrix_create(BATCH, MNIST_PIXELS);
    Matrix *Y = matrix_create(BATCH, CLASSES);

    for (int epoch = 1; epoch <= EPOCHS; epoch++) {
        clock_t t0 = clock();
        shuffle(order, n_train);

        float loss_sum = 0.0f;
        int batches = 0;
        for (int start = 0; start + BATCH <= n_train; start += BATCH) {
            fill_batch(X, Y, train_x, train_y, &order[start], BATCH);
            loss_sum += train_batch(&net, X, Y);
            batches++;
        }

        float acc = accuracy(&net, test_x, test_y, n_test);
        printf("epoch %d/%d   loss = %.4f   test accuracy = %.2f%%   (%.1f s)\n",
               epoch, EPOCHS, loss_sum / (float)batches, acc * 100.0f,
               (double)(clock() - t0) / CLOCKS_PER_SEC);
    }

    matrix_free(X); matrix_free(Y);
    net_free(&net);
    free(order); free(train_x); free(train_y); free(test_x); free(test_y);
    return 0;
}
