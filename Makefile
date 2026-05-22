CC = gcc
CFLAGS = -O3 -std=c11 -D_POSIX_C_SOURCE=200112L -Iinclude -fopenmp
LDFLAGS = -lm
SRCS = src/main.c src/nn.c src/math_utils.c src/math_omp.c src/train_seq.c src/train_omp.c
TARGET = benchmark

.PHONY: all clean

all: $(TARGET)

$(TARGET): $(SRCS)
	$(CC) $(CFLAGS) $(SRCS) -o $(TARGET) $(LDFLAGS)

clean:
	rm -f $(TARGET)
