#ifndef MNIST_H
#define MNIST_H

#define MNIST_PIXELS 784   /* 28 x 28 */

/* Loads an IDX image file. Returns count * 784 floats scaled to 0..1
   (caller must free), or NULL on error. */
float *mnist_load_images(const char *path, int *count);

/* Loads an IDX label file. Returns count bytes with values 0..9
   (caller must free), or NULL on error. */
unsigned char *mnist_load_labels(const char *path, int *count);

#endif
