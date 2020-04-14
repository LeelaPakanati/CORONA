CC=g++

all: sim

sim: main.cpp datatypes/*.cpp
	$(CC) main.cpp -o main

clean:
	rm main
