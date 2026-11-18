/* A small AI that sorts tasks/notes into categories you define.
 *
 *   taskai train [tasks.txt]   learn from examples, save model.tai
 *   taskai chat  [tasks.txt]   type a task, get its category
 *
 * How it works: each text becomes a "bag of words" (a 0/1 vector saying which
 * known words appear), then a small neural network maps that vector to categories.
 */
#include <ctype.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "matrix.h"

#define MAX_LABELS   16
#define LABEL_LEN    32
#define MAX_VOCAB    4000
#define WORD_LEN     24
#define MAX_SAMPLES  5000
#define MAX_LINE     512
#define MAX_TOKENS   64

#define HIDDEN        32
#define EPOCHS        400
#define LEARN_RATE    0.01f
#define WEIGHT_DECAY  0.003f
#define BETA1 0.9f
#define BETA2 0.999f
#define ADAM_EPS 1e-8f
#define MODEL_FILE  "model.tai"
#define MAGIC 0x54414931

/* ---------------- data ---------------- */

typedef struct { char *text; int label; } Sample;

typedef struct {
    Sample samples[MAX_SAMPLES];
    int n;
    char labels[MAX_LABELS][LABEL_LEN];
    int n_labels;
} Dataset;

typedef struct {
    char words[MAX_VOCAB][WORD_LEN];
    int n;
} Vocab;

/* Split text into lowercase words (letters/digits, length >= 2). */
static int tokenize(const char *text, char tokens[][WORD_LEN], int max_tokens) {
    int n = 0, len = 0;
    char cur[WORD_LEN];
    for (const char *p = text;; p++) {
        unsigned char c = (unsigned char)*p;
        if (c < 128 && isalnum(c)) {
            if (len < WORD_LEN - 1) cur[len++] = (char)tolower(c);
        } else {
            if (len >= 2 && n < max_tokens) {
                cur[len] = 0;
                strcpy(tokens[n++], cur);
            }
            len = 0;
            if (c == 0) break;
        }
    }
    return n;
}

static int vocab_find(const Vocab *v, const char *w) {
    for (int i = 0; i < v->n; i++)
        if (strcmp(v->words[i], w) == 0) return i;
    return -1;
}

static void vocab_build(Vocab *v, const Sample *s, const int *idx, int count) {
    char tokens[MAX_TOKENS][WORD_LEN];
    v->n = 0;
    for (int k = 0; k < count; k++) {
        int nt = tokenize(s[idx[k]].text, tokens, MAX_TOKENS);
        for (int t = 0; t < nt; t++)
            if (vocab_find(v, tokens[t]) < 0 && v->n < MAX_VOCAB)
                strcpy(v->words[v->n++], tokens[t]);
    }
}

/* Mark which known words appear in text. Returns how many known words were found. */
static int featurize(Matrix *X, int row, const char *text, const Vocab *v) {
    char tokens[MAX_TOKENS][WORD_LEN];
    int nt = tokenize(text, tokens, MAX_TOKENS), known = 0;
    for (int t = 0; t < nt; t++) {
        int i = vocab_find(v, tokens[t]);
        if (i >= 0) { MAT_AT(X, row, i) = 1.0f; known++; }
    }
    return known;
}

/* Our own strdup: strdup is not part of standard C11. */
static char *copy_string(const char *s) {
    size_t n = strlen(s) + 1;
    char *c = malloc(n);
    if (c) memcpy(c, s, n);
    return c;
}

static void trim(char *s) {
    size_t n = strlen(s);
    while (n > 0 && isspace((unsigned char)s[n - 1])) s[--n] = 0;
    size_t start = 0;
    while (s[start] && isspace((unsigned char)s[start])) start++;
    if (start) memmove(s, s + start, strlen(s + start) + 1);
}

static int load_data(const char *path, Dataset *d) {
    FILE *f = fopen(path, "r");
    if (!f) { fprintf(stderr, "Cannot open %s\n", path); return 0; }

    char line[MAX_LINE];
    d->n = 0; d->n_labels = 0;
    while (fgets(line, sizeof line, f)) {
        char *l = line;
        if ((unsigned char)l[0] == 0xEF && (unsigned char)l[1] == 0xBB && (unsigned char)l[2] == 0xBF)
            l += 3;                                   /* skip a UTF-8 BOM from Notepad */
        trim(l);
        if (l[0] == 0 || l[0] == '#') continue;

        char *bar = strchr(l, '|');
        if (!bar) { fprintf(stderr, "Skipping line without '|': %s\n", l); continue; }
        *bar = 0;
        char *label = l, *text = bar + 1;
        trim(label); trim(text);
        if (!label[0] || !text[0]) continue;

        int li = -1;
        for (int i = 0; i < d->n_labels; i++)
            if (strcmp(d->labels[i], label) == 0) li = i;
        if (li < 0) {
            if (d->n_labels >= MAX_LABELS) { fprintf(stderr, "Too many categories (max %d)\n", MAX_LABELS); continue; }
            snprintf(d->labels[d->n_labels], LABEL_LEN, "%.31s", label);
            li = d->n_labels++;
        }
        if (d->n >= MAX_SAMPLES) break;
        d->samples[d->n].text = copy_string(text);
        d->samples[d->n].label = li;
        d->n++;
    }
    fclose(f);
    return 1;
}

static void free_data(Dataset *d) {
    for (int i = 0; i < d->n; i++) free(d->samples[i].text);
}

/* ---------------- the network (same ideas as Phase 5) ---------------- */

typedef struct { Matrix *w, *g, *m, *v; } Param;
typedef struct { Param W1, b1, W2, b2; int t; } Net;

static float relu(float z) { return z > 0.0f ? z : 0.0f; }

static Param param_create(int r, int c) {
    Param p = { matrix_create(r, c), matrix_create(r, c), matrix_create(r, c), matrix_create(r, c) };
    return p;
}
static void param_free(Param *p) {
    matrix_free(p->w); matrix_free(p->g); matrix_free(p->m); matrix_free(p->v);
}
static void param_set_grad(Param *p, Matrix *g) { matrix_free(p->g); p->g = g; }

static void adam_step(Param *p, int t) {
    float c1 = 1.0f - powf(BETA1, (float)t), c2 = 1.0f - powf(BETA2, (float)t);
    for (int i = 0; i < p->w->rows * p->w->cols; i++) {
        float g = p->g->data[i];
        p->m->data[i] = BETA1 * p->m->data[i] + (1.0f - BETA1) * g;
        p->v->data[i] = BETA2 * p->v->data[i] + (1.0f - BETA2) * g * g;
        p->w->data[i] -= LEARN_RATE * (p->m->data[i] / c1) / (sqrtf(p->v->data[i] / c2) + ADAM_EPS);
    }
}

static void add_bias(Matrix *m, const Matrix *b) {
    for (int i = 0; i < m->rows; i++)
        for (int j = 0; j < m->cols; j++) MAT_AT(m, i, j) += b->data[j];
}
static Matrix *sum_rows(const Matrix *m) {
    Matrix *s = matrix_create(1, m->cols);
    for (int i = 0; i < m->rows; i++)
        for (int j = 0; j < m->cols; j++) s->data[j] += MAT_AT(m, i, j);
    return s;
}
static void softmax_rows(Matrix *m) {
    for (int i = 0; i < m->rows; i++) {
        float mx = MAT_AT(m, i, 0), sum = 0.0f;
        for (int j = 1; j < m->cols; j++) if (MAT_AT(m, i, j) > mx) mx = MAT_AT(m, i, j);
        for (int j = 0; j < m->cols; j++) { MAT_AT(m, i, j) = expf(MAT_AT(m, i, j) - mx); sum += MAT_AT(m, i, j); }
        for (int j = 0; j < m->cols; j++) MAT_AT(m, i, j) /= sum;
    }
}

static Net net_create(int V, int C) {
    Net n;
    n.W1 = param_create(V, HIDDEN);  n.b1 = param_create(1, HIDDEN);
    n.W2 = param_create(HIDDEN, C);  n.b2 = param_create(1, C);
    n.t = 0;
    float l1 = sqrtf(6.0f / (float)V), l2 = sqrtf(6.0f / HIDDEN);
    matrix_randomize(n.W1.w, -l1, l1);
    matrix_randomize(n.W2.w, -l2, l2);
    return n;
}
static void net_free(Net *n) {
    param_free(&n->W1); param_free(&n->b1); param_free(&n->W2); param_free(&n->b2);
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

/* One full-batch training step. Returns the loss. */
static float train_step(Net *n, const Matrix *X, const Matrix *Y) {
    int B = X->rows;
    Matrix *A1, *P;
    forward(n, X, &A1, &P);

    float loss = 0.0f;
    for (int i = 0; i < B; i++)
        for (int j = 0; j < Y->cols; j++)
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
        if (A1->data[i] <= 0.0f) dZ1->data[i] = 0.0f;

    Matrix *XT = matrix_transpose(X);
    param_set_grad(&n->W1, matrix_mul(XT, dZ1));
    param_set_grad(&n->b1, sum_rows(dZ1));

    /* Weight decay: gently pulls weights toward 0 so the net doesn't just memorize. */
    for (int i = 0; i < n->W1.w->rows * n->W1.w->cols; i++) n->W1.g->data[i] += WEIGHT_DECAY * n->W1.w->data[i];
    for (int i = 0; i < n->W2.w->rows * n->W2.w->cols; i++) n->W2.g->data[i] += WEIGHT_DECAY * n->W2.w->data[i];

    n->t++;
    adam_step(&n->W1, n->t); adam_step(&n->b1, n->t);
    adam_step(&n->W2, n->t); adam_step(&n->b2, n->t);

    matrix_free(A1); matrix_free(P); matrix_free(dZ2); matrix_free(A1T);
    matrix_free(W2T); matrix_free(dZ1); matrix_free(XT);
    return loss;
}

static int argmax_row(const Matrix *P, int row) {
    int best = 0;
    for (int j = 1; j < P->cols; j++) if (MAT_AT(P, row, j) > MAT_AT(P, row, best)) best = j;
    return best;
}

static float accuracy(const Net *n, const Matrix *X, const int *labels) {
    Matrix *A1, *P;
    forward(n, X, &A1, &P);
    int ok = 0;
    for (int i = 0; i < X->rows; i++) if (argmax_row(P, i) == labels[i]) ok++;
    matrix_free(A1); matrix_free(P);
    return (float)ok / (float)X->rows;
}

/* ---------------- saving / loading ---------------- */

static int save_model(const Net *n, const Vocab *v, const Dataset *d, const char *path) {
    FILE *f = fopen(path, "wb");
    if (!f) return 0;
    int magic = MAGIC, V = v->n, C = d->n_labels;
    fwrite(&magic, sizeof(int), 1, f); fwrite(&V, sizeof(int), 1, f); fwrite(&C, sizeof(int), 1, f);
    fwrite(d->labels, LABEL_LEN, (size_t)C, f);
    fwrite(v->words, WORD_LEN, (size_t)V, f);
    const Matrix *ms[4] = { n->W1.w, n->b1.w, n->W2.w, n->b2.w };
    for (int k = 0; k < 4; k++)
        fwrite(ms[k]->data, sizeof(float), (size_t)ms[k]->rows * ms[k]->cols, f);
    fclose(f);
    return 1;
}

static int load_model(const char *path, Net *n, Vocab *v, Dataset *d) {
    FILE *f = fopen(path, "rb");
    if (!f) return 0;
    int magic, V, C;
    if (fread(&magic, sizeof(int), 1, f) != 1 || magic != MAGIC ||
        fread(&V, sizeof(int), 1, f) != 1 || fread(&C, sizeof(int), 1, f) != 1 ||
        V <= 0 || V > MAX_VOCAB || C <= 0 || C > MAX_LABELS) { fclose(f); return 0; }
    d->n_labels = C; v->n = V; d->n = 0;
    if (fread(d->labels, LABEL_LEN, (size_t)C, f) != (size_t)C ||
        fread(v->words, WORD_LEN, (size_t)V, f) != (size_t)V) { fclose(f); return 0; }
    *n = net_create(V, C);
    Matrix *ms[4] = { n->W1.w, n->b1.w, n->W2.w, n->b2.w };
    for (int k = 0; k < 4; k++) {
        size_t cnt = (size_t)ms[k]->rows * ms[k]->cols;
        if (fread(ms[k]->data, sizeof(float), cnt, f) != cnt) { fclose(f); net_free(n); return 0; }
    }
    fclose(f);
    return 1;
}

/* ---------------- commands ---------------- */

static void shuffle(int *a, int n) {
    for (int i = n - 1; i > 0; i--) { int j = rand() % (i + 1); int t = a[i]; a[i] = a[j]; a[j] = t; }
}

static int cmd_train(const char *data_path, int quiet) {
    static Dataset d;
    static Vocab vocab;
    if (!load_data(data_path, &d)) return 1;
    if (!quiet) {
        printf("Loaded %d examples in %d categories:", d.n, d.n_labels);
        for (int i = 0; i < d.n_labels; i++) printf(" %s", d.labels[i]);
        printf("\n");
    }
    if (d.n < 20 || d.n_labels < 2) { fprintf(stderr, "Need at least 20 examples and 2 categories.\n"); return 1; }

    /* Hold out 20% of the examples to test on. The AI never trains on these. */
    srand(42);
    int *order = malloc((size_t)d.n * sizeof(int));
    for (int i = 0; i < d.n; i++) order[i] = i;
    shuffle(order, d.n);
    int n_val = d.n / 5, n_train = d.n - n_val;
    int *tr = order, *va = order + n_train;

    vocab_build(&vocab, d.samples, tr, n_train);   /* vocabulary from training texts only */
    int V = vocab.n, C = d.n_labels;
    if (!quiet) printf("Vocabulary: %d words. Training on %d, testing on %d.\n\n", V, n_train, n_val);

    Matrix *Xt = matrix_create(n_train, V), *Yt = matrix_create(n_train, C);
    Matrix *Xv = matrix_create(n_val, V);
    int *lt = malloc((size_t)n_train * sizeof(int)), *lv = malloc((size_t)n_val * sizeof(int));
    for (int i = 0; i < n_train; i++) {
        featurize(Xt, i, d.samples[tr[i]].text, &vocab);
        lt[i] = d.samples[tr[i]].label;
        MAT_AT(Yt, i, lt[i]) = 1.0f;
    }
    for (int i = 0; i < n_val; i++) {
        featurize(Xv, i, d.samples[va[i]].text, &vocab);
        lv[i] = d.samples[va[i]].label;
    }

    Net net = net_create(V, C);
    for (int e = 1; e <= EPOCHS; e++) {
        float loss = train_step(&net, Xt, Yt);
        if (!quiet && (e % 100 == 0 || e == 1))
            printf("epoch %3d   loss = %.4f   train accuracy = %5.1f%%   test accuracy = %5.1f%%\n",
                   e, loss, accuracy(&net, Xt, lt) * 100.0f, accuracy(&net, Xv, lv) * 100.0f);
    }

    float final_val = accuracy(&net, Xv, lv);

    if (!quiet) {
        /* Show what it got wrong on the held-out examples. */
        Matrix *A1, *P;
        forward(&net, Xv, &A1, &P);
        printf("\nMistakes on the test examples:\n");
        int wrong = 0;
        for (int i = 0; i < n_val; i++) {
            int g = argmax_row(P, i);
            if (g != lv[i]) {
                printf("  \"%s\"\n      real: %s   guessed: %s\n", d.samples[va[i]].text, d.labels[lv[i]], d.labels[g]);
                wrong++;
            }
        }
        if (!wrong) printf("  none\n");
        matrix_free(A1); matrix_free(P);
    }

    if (!save_model(&net, &vocab, &d, MODEL_FILE)) fprintf(stderr, "Could not write %s\n", MODEL_FILE);
    else if (quiet) printf("Retrained on %d examples (%d categories). Test accuracy: %.1f%%\n", d.n, C, final_val * 100.0f);
    else printf("\nSaved %s. Now run:  taskai chat\n", MODEL_FILE);

    matrix_free(Xt); matrix_free(Yt); matrix_free(Xv);
    free(lt); free(lv); free(order);
    net_free(&net); free_data(&d);
    return 0;
}

static int cmd_chat(const char *data_path) {
    static Dataset d;
    static Vocab vocab;
    Net net;
    if (!load_model(MODEL_FILE, &net, &vocab, &d)) {
        fprintf(stderr, "No usable %s found. Run:  taskai train\n", MODEL_FILE);
        return 1;
    }
    printf("Task sorter ready (%d categories:", d.n_labels);
    for (int i = 0; i < d.n_labels; i++) printf(" %s", d.labels[i]);
    printf(").\nType a task. Commands: /teach category|text   /retrain   /help   /quit\n\n");

    char line[MAX_LINE];
    for (;;) {
        printf("> ");
        fflush(stdout);
        if (!fgets(line, sizeof line, stdin)) break;
        trim(line);
        if (!line[0]) continue;
        if (strcmp(line, "/quit") == 0) break;

        if (strcmp(line, "/help") == 0) {
            printf("  Type any task to sort it.\n"
                   "  /teach category|text   add a correct example to %s\n"
                   "  /retrain               learn from %s now (takes a few seconds)\n"
                   "  /quit                  leave\n", data_path, data_path);
            continue;
        }

        if (strcmp(line, "/retrain") == 0) {
            printf("Retraining...\n");
            if (cmd_train(data_path, 1) == 0) {
                net_free(&net);
                if (!load_model(MODEL_FILE, &net, &vocab, &d)) {
                    fprintf(stderr, "Could not reload %s\n", MODEL_FILE);
                    return 1;
                }
                printf("Ready. Try your tasks again.\n");
            }
            continue;
        }

        if (strncmp(line, "/teach ", 7) == 0) {
            if (!strchr(line + 7, '|')) { printf("Use:  /teach category|task text\n"); continue; }
            FILE *f = fopen(data_path, "a");
            if (!f) { printf("Cannot open %s\n", data_path); continue; }
            fprintf(f, "%s\n", line + 7);
            fclose(f);
            printf("Added to %s. Type /retrain to learn it.\n", data_path);
            continue;
        }

        Matrix *X = matrix_create(1, vocab.n);
        int known = featurize(X, 0, line, &vocab);
        if (known == 0) {
            printf("  I don't know any of those words yet. Use /teach to add an example.\n");
            matrix_free(X);
            continue;
        }
        Matrix *A1, *P;
        forward(&net, X, &A1, &P);

        int top[3] = { -1, -1, -1 };
        for (int k = 0; k < 3 && k < d.n_labels; k++) {
            int best = -1;
            for (int j = 0; j < d.n_labels; j++) {
                int used = 0;
                for (int q = 0; q < k; q++) if (top[q] == j) used = 1;
                if (!used && (best < 0 || P->data[j] > P->data[best])) best = j;
            }
            top[k] = best;
        }
        if (P->data[top[0]] < 0.5f) printf("  Not sure. Best guesses:\n");
        for (int k = 0; k < 3 && top[k] >= 0; k++)
            printf("  %s%-10s %5.1f%%\n", k == 0 ? "-> " : "   ", d.labels[top[k]], P->data[top[k]] * 100.0f);

        matrix_free(X); matrix_free(A1); matrix_free(P);
    }
    net_free(&net);
    return 0;
}

int main(int argc, char **argv) {
    const char *data_path = (argc > 2) ? argv[2] : "tasks.txt";
    if (argc > 1 && strcmp(argv[1], "train") == 0) return cmd_train(data_path, 0);
    if (argc > 1 && strcmp(argv[1], "chat") == 0)  return cmd_chat(data_path);
    printf("Usage:\n  taskai train [tasks.txt]\n  taskai chat  [tasks.txt]\n");
    return 1;
}
