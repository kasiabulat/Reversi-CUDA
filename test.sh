#!/bin/bash

g++ -c -Wall -std=c++11 board_test.cpp board.cpp board_factory.cpp
#g++ -c -Wall board.cpp 
#g++ -c -Wall board_factory.cpp
g++ -Wall -o board_test.exe board_test.o board.o board_factory.o
./board_test.exe
