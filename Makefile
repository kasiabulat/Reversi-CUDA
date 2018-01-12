CUDA_INSTALL_PATH ?= /usr/local/cuda
VER = 

CXX := /usr/bin/g++$(VER)
CC := /usr/bin/gcc$(VER)
LINK := /usr/bin/g++$(VER) -fPIC
CCPATH := ./gcc$(VER)
NVCC  := $(CUDA_INSTALL_PATH)/bin/nvcc -lcurand -ccbin $(CCPATH)

# Includes
INCLUDES = -I. -I$(CUDA_INSTALL_PATH)/include

# Libraries
LIB_CUDA := -L/usr/lib/nvidia-current -lcuda

# Options
NVCCOPTIONS = -ptx -Wno-deprecated-gpu-targets
CXXOPTIONS = -std=c++17 -O2 -lcurand

# Common flags
COMMONFLAGS += $(INCLUDES) -lcurand
NVCCFLAGS += $(COMMONFLAGS) $(NVCCOPTIONS)
CXXFLAGS += $(COMMONFLAGS) $(CXXOPTIONS)
CFLAGS += $(COMMONFLAGS) -lcurand

CUDA_OBJS = randomized_play_player.ptx 
OBJS = demo.cpp.o randomized_play_player.cpp.o board.cpp.o board_factory.cpp.o
TARGET = solution.x
LINKLINE =  $(LINK) -o $(TARGET) $(OBJS) $(LIB_CUDA)

.SUFFIXES:	.c	.cpp	.cu	.o	
%.c.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

%.ptx: %.cu
	$(NVCC) $(NVCCFLAGS) $< -o $@ 

%.cpp.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@ 

$(TARGET): prepare $(OBJS) $(CUDA_OBJS)
	$(LINKLINE)

clean:
	rm -rf $(TARGET) *.o *.ptx

prepare:
	rm -rf $(CCPATH);\
	mkdir -p $(CCPATH);\
	ln -s $(CXX) $(CCPATH)/g++;\
	ln -s $(CC) $(CCPATH)/gcc;

