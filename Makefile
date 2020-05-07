CC=g++
CUDA_CC=nvcc

all: cpp cuda

debug: main.cpp datatypes/*.cpp
	$(CC) main.cpp -g -O0 -o main

cpp: main.cpp datatypes/*.cpp
	$(CC) main.cpp -o main

cuda: main.cpp datatypes/*.cpp
	$(CUDA_CC) main.cpp -o main_cuda

clean:
	rm main
