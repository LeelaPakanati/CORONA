CC=g++

all: sim

debug: main.cpp datatypes/*.cpp
	$(CC) main.cpp -g -O0 -o main

sim: main.cpp datatypes/*.cpp
	$(CC) main.cpp -o main

clean:
	rm main
