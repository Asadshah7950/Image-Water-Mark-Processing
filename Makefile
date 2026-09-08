CC = gcc
MPICC = mpicc
NVCC = nvcc
CFLAGS = -O3 -Wall
MPIFLAGS = -O3 -Wall
NVCCFLAGS = -O3 -arch=sm_75
LDFLAGS = -lm

all: dirs bin/watermark

dirs:
\tmkdir -p build bin results uploads

build/image_utils.o: src/image_utils.c
\t$(CC) $(CFLAGS) -Iinclude -c -o $@ $<

build/watermark_cpu.o: src/watermark_cpu.c
\t$(CC) $(CFLAGS) -Iinclude -c -o $@ $<

build/watermark_cuda.o: src/watermark_cuda.cu
\t$(NVCC) $(NVCCFLAGS) -Iinclude -c -o $@ $<

build/main.o: src/main.c
\t$(MPICC) $(MPIFLAGS) -Iinclude -c -o $@ $<

bin/watermark: build/main.o build/image_utils.o build/watermark_cuda.o build/watermark_cpu.o
\t$(MPICC) -o $@ $^ $(LDFLAGS) -L/usr/local/cuda/lib64 -lcudart

clean:
\trm -rf build bin results

. PHONY: all dirs clean
