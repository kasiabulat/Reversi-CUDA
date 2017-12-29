#!/bin/bash

g++ -c -Wall board_test.cpp
g++ -c -Wall board.cpp
g++ -c -Wall board_factory.cpp
g++ -Wall -o board_test.exe board_test.o board.o board_factory.o
./board_test.exe
