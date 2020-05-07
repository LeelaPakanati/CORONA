CC=g++
CUDA_CC=nvcc

all: cpp cuda

debug: cpu-debug cuda-debug

cpu-debug: main.cpp datatypes/*.cpp datatypes/*.h
	$(CC) main.cpp -g -O0 -o main

cpp: main.cpp datatypes/*.cpp datatypes/*.h
	$(CC) main.cpp -o main

cuda: main.cpp datatypes/*.cpp datatypes/*.h
	$(CUDA_CC) main.cu -o main_cuda

cuda-debug: main.cu datatypes/*.cpp datatypes/*.h
	$(CUDA_CC) main.cu -g -G -O0 -o main

clean:
	rm main
