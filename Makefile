CC = gcc
MPICC = mpicc
NVCC = nvcc
CFLAGS = -O3 -Wall
MPIFLAGS = -O3 -Wall
NVCCFLAGS = -O3 -arch=sm_75
LDFLAGS = -lm

all: dirs bin/watermark

dirs:
	mkdir -p build bin results uploads

build/image_utils.o: src/image_utils.c
	$(CC) $(CFLAGS) -Iinclude -c -o $@ $<

build/watermark_cpu.o: src/watermark_cpu.c
	$(CC) $(CFLAGS) -Iinclude -c -o $@ $<

build/watermark_cuda.o: src/watermark_cuda.cu
	$(NVCC) $(NVCCFLAGS) -Iinclude -c -o $@ $<

build/main.o: src/main.c
	$(MPICC) $(MPIFLAGS) -Iinclude -c -o $@ $<

bin/watermark: build/main.o build/image_utils.o build/watermark_cuda.o build/watermark_cpu.o
	$(MPICC) -o $@ $^ $(LDFLAGS) -L/usr/local/cuda/lib64 -lcudart

bin/test_runner: dirs build/image_utils.o build/watermark_cpu.o test/test_runner.c
	$(CC) $(CFLAGS) -Iinclude src/image_utils.c src/watermark_cpu.c test/test_runner.c -o $@ $(LDFLAGS)

test: bin/test_runner
	./bin/test_runner

clean:
	rm -rf build bin results

.PHONY: all dirs test clean
