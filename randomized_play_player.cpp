#include "cuda.h"
#include <cstdio>
#include <iostream>
#include <cstring>
#include <stdlib.h> 
#include <list> 

#include "board.h"
#include "randomized_play_player.h"

using namespace std;

const char *cuda_error_string(CUresult result) { 
    switch(result) { 
    case CUDA_SUCCESS: return "No errors"; 
    case CUDA_ERROR_INVALID_VALUE: return "Invalid value"; 
    case CUDA_ERROR_OUT_OF_MEMORY: return "Out of memory"; 
    case CUDA_ERROR_NOT_INITIALIZED: return "Driver not initialized"; 
    case CUDA_ERROR_DEINITIALIZED: return "Driver deinitialized"; 

    case CUDA_ERROR_NO_DEVICE: return "No CUDA-capable device available"; 
    case CUDA_ERROR_INVALID_DEVICE: return "Invalid device"; 

    case CUDA_ERROR_INVALID_IMAGE: return "Invalid kernel image"; 
    case CUDA_ERROR_INVALID_CONTEXT: return "Invalid context"; 
    case CUDA_ERROR_CONTEXT_ALREADY_CURRENT: return "Context already current"; 
    case CUDA_ERROR_MAP_FAILED: return "Map failed"; 
    case CUDA_ERROR_UNMAP_FAILED: return "Unmap failed"; 
    case CUDA_ERROR_ARRAY_IS_MAPPED: return "Array is mapped"; 
    case CUDA_ERROR_ALREADY_MAPPED: return "Already mapped"; 
    case CUDA_ERROR_NO_BINARY_FOR_GPU: return "No binary for GPU"; 
    case CUDA_ERROR_ALREADY_ACQUIRED: return "Already acquired"; 
    case CUDA_ERROR_NOT_MAPPED: return "Not mapped"; 
    case CUDA_ERROR_NOT_MAPPED_AS_ARRAY: return "Mapped resource not available for access as an array"; 
    case CUDA_ERROR_NOT_MAPPED_AS_POINTER: return "Mapped resource not available for access as a pointer"; 
    case CUDA_ERROR_ECC_UNCORRECTABLE: return "Uncorrectable ECC error detected"; 
    case CUDA_ERROR_UNSUPPORTED_LIMIT: return "CUlimit not supported by device";    

    case CUDA_ERROR_INVALID_SOURCE: return "Invalid source"; 
    case CUDA_ERROR_FILE_NOT_FOUND: return "File not found"; 
    case CUDA_ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND: return "Link to a shared object failed to resolve"; 
    case CUDA_ERROR_SHARED_OBJECT_INIT_FAILED: return "Shared object initialization failed"; 

    case CUDA_ERROR_INVALID_HANDLE: return "Invalid handle"; 

    case CUDA_ERROR_NOT_FOUND: return "Not found"; 

    case CUDA_ERROR_NOT_READY: return "CUDA not ready"; 

    case CUDA_ERROR_LAUNCH_FAILED: return "Launch failed"; 
    case CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES: return "Launch exceeded resources"; 
    case CUDA_ERROR_LAUNCH_TIMEOUT: return "Launch exceeded timeout"; 
    case CUDA_ERROR_LAUNCH_INCOMPATIBLE_TEXTURING: return "Launch with incompatible texturing"; 

    case CUDA_ERROR_UNKNOWN: return "Unknown error"; 

    default: return "Unknown CUDA error value"; 
    } 
}

void print_error(string error) {
	cout << error;
	exit(1);
}

RandomizedPlayPlayerCuda::RandomizedPlayPlayerCuda(string name, int seed, int number_of_tries){
	this->name = name;
	this->seed = seed;
	this->number_of_tries = number_of_tries;
}

int RandomizedPlayPlayerCuda::make_move(Board board) {		
	/// load module
    CUmodule cuModuleRandomizedPlayPlayer = (CUmodule)0;
    if(cuModuleLoad(&cuModuleRandomizedPlayPlayer, "randomized_play_player.ptx") != CUDA_SUCCESS) 
        print_error("cannot load module randomized_play_player\n");  
 
    /// kernel functions
    CUfunction check_move;
    if(cuModuleGetFunction(&check_move, cuModuleRadius, "check_move") != CUDA_SUCCESS) 
        print_error("cannot acquire kernel handle for check_move\n");
		
	vector<int> correct_moves;	
	for(int i=0; i<64; i++) 
		if(board.is_correct_move(i) 
			correct_moves.push_back(i);
			
	unsigned int blockDimX = 1024;
	unsigned int gridDimX = (this->number_of_tries + blockDimX - 1)/blockDimX;		
	
	vector<pair<int,int> > moves_values; 
	
	/// calculate moves values using CUDA
	for(auto move : correct_moves) {
		auto board_to_check = board.make_move(move);
		int result = 0;
		int tries = this->number_of_tries;
		CUdeviceptr result_array_device;
	    if(cuMemAlloc(&result_array_device, this->number_of_tries*sizeof(int)) != CUDA_SUCCESS) {
	    	print_error("failed to alloc sth_changed\n");
		}
		
		void* args[] = {&seed, &board_to_check.player_pieces, &board_to_check.opponent_pieces, &result_array_device};
    	if(cuLaunchKernel(check_move, gridDimX, 1, 1, blockDimX, 1, 1, 0, 0, args, 0) != CUDA_SUCCESS) 
	    	print_error("cannot run kernel check_move\n");
	
		if (cuCtxSynchronize() != CUDA_SUCCESS) 
        	print_error("failed to synchronize\n");
        
        int * values = (int*) malloc(this->number_of_tries*sizeof(int));
        if (cuMemcpyDtoH(values,result_array_device,this->number_of_tries*sizeof(int)) != CUDA_SUCCESS) 
        	print_error("failed to copy result from result_array_device");
        
        long long sum = 0;
        for(auto value in values) 
        	sum += value;
		
        moves_values.push_back(make_pair(sum,move));
	}
	
	/// find the best move
	sort(moves_values.begin(), moves_values.end());
	return moves_values.back().second;
}


