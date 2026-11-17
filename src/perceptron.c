/* Phase 2: a single neuron that learns logic gates.
 *
 *   inputs x1, x2  ->  z = w1*x1 + w2*x2 + b  ->  output = sigmoid(z)
 *
 * Learning = repeatedly nudging w1, w2, b to reduce the error.
 */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define N_SAMPLES 4

typedef struct {
    float w[2];
    float b;
} Neuron;

static float sigmoid(float z) { return 1.0f / (1.0f + expf(-z)); }

/* Forward pass: make a prediction between 0 and 1. */
static float predict(const Neuron *n, const float x[2]) {
    return sigmoid(n->w[0] * x[0] + n->w[1] * x[1] + n->b);
}

/* Random value in [-1, 1] for starting weights. */
static float rand_weight(void) {
    return 2.0f * ((float)rand() / (float)RAND_MAX) - 1.0f;
}

static void train(const char *name, const float inputs[N_SAMPLES][2],
                  const float targets[N_SAMPLES], int epochs, float lr) {
    Neuron n = { { rand_weight(), rand_weight() }, rand_weight() };

    printf("=== %s ===\n", name);

    for (int epoch = 0; epoch <= epochs; epoch++) {
        float loss = 0.0f;

        for (int s = 0; s < N_SAMPLES; s++) {
            const float *x = inputs[s];
            float out = predict(&n, x);
            float err = out - targets[s];
            loss += err * err;                        /* squared error */

            /* Gradient via the chain rule:
               dLoss/dz = err * sigmoid'(z) = err * out * (1 - out) */
            float dz = err * out * (1.0f - out);

            n.w[0] -= lr * dz * x[0];                 /* dz/dw1 = x1 */
            n.w[1] -= lr * dz * x[1];                 /* dz/dw2 = x2 */
            n.b    -= lr * dz;                        /* dz/db  = 1  */
        }

        if (epoch % (epochs / 5) == 0)
            printf("  epoch %5d   loss = %.5f\n", epoch, loss / N_SAMPLES);
    }

    printf("  results:\n");
    int correct = 0;
    for (int s = 0; s < N_SAMPLES; s++) {
        float out = predict(&n, inputs[s]);
        int guess = out >= 0.1f;
        if (guess == (int)targets[s]) correct++;
        printf("    %g op %g -> %.3f (rounded %d, want %g)\n",
               inputs[s][0], inputs[s][1], out, guess, targets[s]);
    }
    printf("  accuracy: %d/%d\n\n", correct, N_SAMPLES);
}

int main(void) {
    srand((unsigned)time(NULL));

    const float inputs[N_SAMPLES][2] = { {0, 0}, {0, 1}, {1, 0}, {1, 1} };

    const float and_t[N_SAMPLES] = { 0, 0, 0, 1 };
    const float or_t[N_SAMPLES]  = { 0, 1, 1, 1 };
    const float xor_t[N_SAMPLES] = { 0, 1, 1, 0 };

    train("AND", inputs, and_t, 10000, 0.5f);
    train("OR",  inputs, or_t,  10000, 0.5f);
    train("XOR (one neuron cannot solve this)", inputs, xor_t, 10000, 0.5f);

    return 0;
}
