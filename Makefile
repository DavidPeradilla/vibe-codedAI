CC      = gcc
CFLAGS  = -std=c11 -Wall -Wextra -O2 -g -Isrc
LDFLAGS = -lm

all: test

build/test_matrix: tests/test_matrix.c src/matrix.c src/matrix.h
	@mkdir -p build
	$(CC) $(CFLAGS) tests/test_matrix.c src/matrix.c -o $@ $(LDFLAGS)

build/perceptron: src/perceptron.c
	@mkdir -p build
	$(CC) $(CFLAGS) src/perceptron.c -o $@ $(LDFLAGS)

test: build/test_matrix
	./build/test_matrix

perceptron: build/perceptron
	./build/perceptron

clean:
	rm -rf build
