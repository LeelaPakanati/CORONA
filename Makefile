CC=g++

all: sim

dbg: main.cpp datatypes/*.cpp
	$(CC) main.cpp -g -o main

sim: main.cpp datatypes/*.cpp
	$(CC) main.cpp -o main

clean:
	rm main
