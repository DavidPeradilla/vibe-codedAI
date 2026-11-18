#include "mnist.h"

#include <stdio.h>
#include <stdlib.h>

/* IDX files store numbers in big-endian byte order, so we assemble them by hand. */
static int read_int32_be(FILE *f, int *out) {
    unsigned char b[4];
    if (fread(b, 1, 4, f) != 4) return 0;
    *out = (b[0] << 24) | (b[1] << 16) | (b[2] << 8) | b[3];
    return 1;
}

float *mnist_load_images(const char *path, int *count) {
    FILE *f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "Cannot open %s\n", path);
        return NULL;
    }

    int magic, n, rows, cols;
    if (!read_int32_be(f, &magic) || !read_int32_be(f, &n) ||
        !read_int32_be(f, &rows) || !read_int32_be(f, &cols) ||
        magic != 2051 || rows * cols != MNIST_PIXELS || n <= 0) {
        fprintf(stderr, "%s is not a valid MNIST image file\n", path);
        fclose(f);
        return NULL;
    }

    size_t total = (size_t)n * MNIST_PIXELS;
    unsigned char *raw = malloc(total);
    float *images = malloc(total * sizeof(float));
    if (!raw || !images || fread(raw, 1, total, f) != total) {
        fprintf(stderr, "Failed reading pixels from %s\n", path);
        free(raw); free(images); fclose(f);
        return NULL;
    }
    fclose(f);

    for (size_t i = 0; i < total; i++) images[i] = raw[i] / 255.0f;
    free(raw);

    *count = n;
    return images;
}

unsigned char *mnist_load_labels(const char *path, int *count) {
    FILE *f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "Cannot open %s\n", path);
        return NULL;
    }

    int magic, n;
    if (!read_int32_be(f, &magic) || !read_int32_be(f, &n) || magic != 2049 || n <= 0) {
        fprintf(stderr, "%s is not a valid MNIST label file\n", path);
        fclose(f);
        return NULL;
    }

    unsigned char *labels = malloc((size_t)n);
    if (!labels || fread(labels, 1, (size_t)n, f) != (size_t)n) {
        fprintf(stderr, "Failed reading labels from %s\n", path);
        free(labels); fclose(f);
        return NULL;
    }
    fclose(f);

    *count = n;
    return labels;
}
