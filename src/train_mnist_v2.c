/* Phase 5: a better MNIST trainer.
 *
 * Changes from Phase 4:
 *   - ReLU instead of sigmoid in the hidden layer (faster, no vanishing gradients)
 *   - He initialization (suited to ReLU)
 *   - Adam optimizer instead of plain gradient descent
 *   - Saves the trained weights to model.bin and reloads them to prove it works
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
#define EPOCHS     10
#define LEARN_RATE 0.001f      /* Adam works well with small rates */
#define BETA1      0.9f
#define BETA2      0.999f
#define ADAM_EPS   1e-8f

/* One trainable matrix plus everything Adam needs to remember about it. */
typedef struct {
    Matrix *w;   /* the parameter itself */
    Matrix *g;   /* gradient from the latest batch */
    Matrix *m;   /* running average of gradients (momentum) */
    Matrix *v;   /* running average of squared gradients */
} Param;

typedef struct {
    Param W1, b1, W2, b2;
    int t;       /* Adam step counter */
} Net;

static float relu(float z) { return z > 0.0f ? z : 0.0f; }

static Param param_create(int rows, int cols) {
    Param p;
    p.w = matrix_create(rows, cols);
    p.g = matrix_create(rows, cols);
    p.m = matrix_create(rows, cols);
    p.v = matrix_create(rows, cols);
    return p;
}

static void param_free(Param *p) {
    matrix_free(p->w); matrix_free(p->g); matrix_free(p->m); matrix_free(p->v);
}

/* Takes ownership of grad (frees it) and stores it in p->g. */
static void param_set_grad(Param *p, Matrix *grad) {
    matrix_free(p->g);
    p->g = grad;
}

/* Adam update for one parameter. */
static void adam_step(Param *p, int t) {
    float c1 = 1.0f - powf(BETA1, (float)t);   /* bias corrections */
    float c2 = 1.0f - powf(BETA2, (float)t);
    for (int i = 0; i < p->w->rows * p->w->cols; i++) {
        float g = p->g->data[i];
        p->m->data[i] = BETA1 * p->m->data[i] + (1.0f - BETA1) * g;
        p->v->data[i] = BETA2 * p->v->data[i] + (1.0f - BETA2) * g * g;
        float mhat = p->m->data[i] / c1;
        float vhat = p->v->data[i] / c2;
        p->w->data[i] -= LEARN_RATE * mhat / (sqrtf(vhat) + ADAM_EPS);
    }
}

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
    n.W1 = param_create(MNIST_PIXELS, HIDDEN);
    n.b1 = param_create(1, HIDDEN);
    n.W2 = param_create(HIDDEN, CLASSES);
    n.b2 = param_create(1, CLASSES);
    n.t = 0;

    /* He initialization: limit = sqrt(6 / number_of_inputs) */
    float lim1 = sqrtf(6.0f / MNIST_PIXELS);
    float lim2 = sqrtf(6.0f / HIDDEN);
    matrix_randomize(n.W1.w, -lim1, lim1);
    matrix_randomize(n.W2.w, -lim2, lim2);
    return n;
}

static void net_free(Net *n) {
    param_free(&n->W1); param_free(&n->b1);
    param_free(&n->W2); param_free(&n->b2);
}

static void forward(const Net *n, const Matrix *X, Matrix **A1, Matrix **P) {
    Matrix *Z1 = matrix_mul(X, n->W1.w);
    add_bias(Z1, n->b1.w);
    *A1 = matrix_apply(Z1, relu);

    *P = matrix_mul(*A1, n->W2.w);
    add_bias(*P, n->b2.w);
    softmax_rows(*P);
    matrix_free(Z1);
}

static void fill_batch(Matrix *X, Matrix *Y, const float *images,
                       const unsigned char *labels, const int *idx, int n) {
    matrix_fill(Y, 0.0f);
    for (int b = 0; b < n; b++) {
        memcpy(&X->data[b * MNIST_PIXELS], &images[(size_t)idx[b] * MNIST_PIXELS],
               MNIST_PIXELS * sizeof(float));
        MAT_AT(Y, b, labels[idx[b]]) = 1.0f;
    }
}

static float train_batch(Net *n, const Matrix *X, const Matrix *Y) {
    int B = X->rows;
    Matrix *A1, *P;
    forward(n, X, &A1, &P);

    float loss = 0.0f;
    for (int i = 0; i < B; i++)
        for (int j = 0; j < CLASSES; j++)
            if (MAT_AT(Y, i, j) > 0.5f) loss -= logf(MAT_AT(P, i, j) + 1e-9f);
    loss /= (float)B;

    Matrix *dZ2 = matrix_sub(P, Y);
    for (int i = 0; i < dZ2->rows * dZ2->cols; i++) dZ2->data[i] /= (float)B;

    Matrix *A1T = matrix_transpose(A1);
    param_set_grad(&n->W2, matrix_mul(A1T, dZ2));
    param_set_grad(&n->b2, sum_rows(dZ2));

    Matrix *W2T = matrix_transpose(n->W2.w);
    Matrix *dZ1 = matrix_mul(dZ2, W2T);
    for (int i = 0; i < dZ1->rows * dZ1->cols; i++)
        if (A1->data[i] <= 0.0f) dZ1->data[i] = 0.0f;     /* ReLU derivative */

    Matrix *XT = matrix_transpose(X);
    param_set_grad(&n->W1, matrix_mul(XT, dZ1));
    param_set_grad(&n->b1, sum_rows(dZ1));

    n->t++;
    adam_step(&n->W1, n->t); adam_step(&n->b1, n->t);
    adam_step(&n->W2, n->t); adam_step(&n->b2, n->t);

    matrix_free(A1); matrix_free(P); matrix_free(dZ2); matrix_free(A1T);
    matrix_free(W2T); matrix_free(dZ1); matrix_free(XT);
    return loss;
}

static float accuracy(const Net *n, const float *images, const unsigned char *labels, int count) {
    const int CHUNK = 500;
    int correct = 0;
    Matrix *X = matrix_create(CHUNK, MNIST_PIXELS);
    for (int start = 0; start < count; start += CHUNK) {
        int m = (count - start < CHUNK) ? count - start : CHUNK;
        memcpy(X->data, &images[(size_t)start * MNIST_PIXELS],
               (size_t)m * MNIST_PIXELS * sizeof(float));
        X->rows = m;
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

/* ---- saving and loading: just rows, cols, then the raw floats, for each matrix ---- */
static int save_model(const Net *n, const char *path) {
    FILE *f = fopen(path, "wb");
    if (!f) return 0;
    const Matrix *ms[4] = { n->W1.w, n->b1.w, n->W2.w, n->b2.w };
    for (int k = 0; k < 4; k++) {
        fwrite(&ms[k]->rows, sizeof(int), 1, f);
        fwrite(&ms[k]->cols, sizeof(int), 1, f);
        fwrite(ms[k]->data, sizeof(float), (size_t)ms[k]->rows * ms[k]->cols, f);
    }
    fclose(f);
    return 1;
}

static int load_model(Net *n, const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) return 0;
    Matrix *ms[4] = { n->W1.w, n->b1.w, n->W2.w, n->b2.w };
    for (int k = 0; k < 4; k++) {
        int r, c;
        if (fread(&r, sizeof(int), 1, f) != 1 || fread(&c, sizeof(int), 1, f) != 1 ||
            r != ms[k]->rows || c != ms[k]->cols ||
            fread(ms[k]->data, sizeof(float), (size_t)r * c, f) != (size_t)r * c) {
            fclose(f);
            return 0;
        }
    }
    fclose(f);
    return 1;
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
        fprintf(stderr, "\nCould not load MNIST from '%s'.\n", dir);
        return 1;
    }
    printf("Loaded %d training and %d test images.\n\n", n_train, n_test);

    Net net = net_create();
    int *order = malloc((size_t)n_train * sizeof(int));
    for (int i = 0; i < n_train; i++) order[i] = i;
    Matrix *X = matrix_create(BATCH, MNIST_PIXELS);
    Matrix *Y = matrix_create(BATCH, CLASSES);

    for (int epoch = 1; epoch <= EPOCHS; epoch++) {
        time_t t0 = time(NULL);
        shuffle(order, n_train);
        float loss_sum = 0.0f;
        int batches = 0;
        for (int start = 0; start + BATCH <= n_train; start += BATCH) {
            fill_batch(X, Y, train_x, train_y, &order[start], BATCH);
            loss_sum += train_batch(&net, X, Y);
            batches++;
        }
        float acc = accuracy(&net, test_x, test_y, n_test);
        printf("epoch %2d/%d   loss = %.4f   test accuracy = %.2f%%   (%ld s)\n",
               epoch, EPOCHS, loss_sum / (float)batches, acc * 100.0f, (long)(time(NULL) - t0));
    }

    /* Save, then load into a brand-new network and check it behaves identically. */
    if (save_model(&net, "model.bin")) {
        Net fresh = net_create();
        if (load_model(&fresh, "model.bin"))
            printf("\nSaved model.bin. Reloaded copy accuracy: %.2f%%\n",
                   accuracy(&fresh, test_x, test_y, n_test) * 100.0f);
        net_free(&fresh);
    } else {
        fprintf(stderr, "Could not write model.bin\n");
    }

    matrix_free(X); matrix_free(Y);
    net_free(&net);
    free(order); free(train_x); free(train_y); free(test_x); free(test_y);
    return 0;
}
