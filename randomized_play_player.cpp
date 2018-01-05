#include <cuda.h>
#include <curand.h>
#include <cstdio>
#include <iostream>
#include <cstring>
#include <cassert>
#include <stdlib.h>
#include <vector>
#include <algorithm>


#include "board.h"
#include "randomized_play_player.h"

using namespace std;

const char*cuda_error_string(CUresult result)
{
	switch(result)
	{
		case CUDA_SUCCESS:
			return "No errors";
		case CUDA_ERROR_INVALID_VALUE:
			return "Invalid value";
		case CUDA_ERROR_OUT_OF_MEMORY:
			return "Out of memory";
		case CUDA_ERROR_NOT_INITIALIZED:
			return "Driver not initialized";
		case CUDA_ERROR_DEINITIALIZED:
			return "Driver deinitialized";

		case CUDA_ERROR_NO_DEVICE:
			return "No CUDA-capable device available";
		case CUDA_ERROR_INVALID_DEVICE:
			return "Invalid device";

		case CUDA_ERROR_INVALID_IMAGE:
			return "Invalid kernel image";
		case CUDA_ERROR_INVALID_CONTEXT:
			return "Invalid context";
		case CUDA_ERROR_CONTEXT_ALREADY_CURRENT:
			return "Context already current";
		case CUDA_ERROR_MAP_FAILED:
			return "Map failed";
		case CUDA_ERROR_UNMAP_FAILED:
			return "Unmap failed";
		case CUDA_ERROR_ARRAY_IS_MAPPED:
			return "Array is mapped";
		case CUDA_ERROR_ALREADY_MAPPED:
			return "Already mapped";
		case CUDA_ERROR_NO_BINARY_FOR_GPU:
			return "No binary for GPU";
		case CUDA_ERROR_ALREADY_ACQUIRED:
			return "Already acquired";
		case CUDA_ERROR_NOT_MAPPED:
			return "Not mapped";
		case CUDA_ERROR_NOT_MAPPED_AS_ARRAY:
			return "Mapped resource not available for access as an array";
		case CUDA_ERROR_NOT_MAPPED_AS_POINTER:
			return "Mapped resource not available for access as a pointer";
		case CUDA_ERROR_ECC_UNCORRECTABLE:
			return "Uncorrectable ECC error detected";
		case CUDA_ERROR_UNSUPPORTED_LIMIT:
			return "CUlimit not supported by device";

		case CUDA_ERROR_INVALID_SOURCE:
			return "Invalid source";
		case CUDA_ERROR_FILE_NOT_FOUND:
			return "File not found";
		case CUDA_ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND:
			return "Link to a shared object failed to resolve";
		case CUDA_ERROR_SHARED_OBJECT_INIT_FAILED:
			return "Shared object initialization failed";

		case CUDA_ERROR_INVALID_HANDLE:
			return "Invalid handle";

		case CUDA_ERROR_NOT_FOUND:
			return "Not found";

		case CUDA_ERROR_NOT_READY:
			return "CUDA not ready";

		case CUDA_ERROR_LAUNCH_FAILED:
			return "Launch failed";
		case CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES:
			return "Launch exceeded resources";
		case CUDA_ERROR_LAUNCH_TIMEOUT:
			return "Launch exceeded timeout";
		case CUDA_ERROR_LAUNCH_INCOMPATIBLE_TEXTURING:
			return "Launch with incompatible texturing";

		case CUDA_ERROR_UNKNOWN:
			return "Unknown error";

		default:
			return "Unknown CUDA error value";
	}
}

void print_error(string error)
{
	cout<<error;
	exit(1);
}

RandomizedPlayPlayerCuda::RandomizedPlayPlayerCuda(string name,int seed,int number_of_tries)
{
	this->name=name;
	this->seed=seed;
	this->number_of_tries=number_of_tries;
}

void check_rand_result(curandStatus_t randStatus,string message="")
{
	if(randStatus!=CURAND_STATUS_SUCCESS)
	{
		cerr<<"Rand status failed: "<<message<<", with status:"<<randStatus<<endl;
		exit(1);
	}
}

void check_result(CUresult result,string message="")
{
	if(result!=CUDA_SUCCESS)
	{
		print_error(message);
		exit(1);
	}
}

int RandomizedPlayPlayerCuda::make_move(Board board)
{
	curandStatus_t randStatus;
	CUresult result;

	/// load module
	CUmodule cuModuleRandomizedPlayPlayer;
	if(cuModuleLoad(&cuModuleRandomizedPlayPlayer,"randomized_play_player.ptx")!=CUDA_SUCCESS)
		print_error("cannot load module randomized_play_player\n");

	/// kernel functions
	CUfunction check_move;
	if(cuModuleGetFunction(&check_move,cuModuleRandomizedPlayPlayer,"check_move")!=CUDA_SUCCESS)
		print_error("cannot acquire kernel handle for check_move\n");



	//vector<int> correct_moves;
	unsigned int correct_moves[64];
	unsigned int numberOfCorrectMoves=0;
	for(int i=0;i<64;i++)
		if(board.is_correct_move(i))
			correct_moves[numberOfCorrectMoves++]=i;
	assert(number_of_tries%1024==0);
	unsigned int blockDimX=1024;
	unsigned int gridDimX=(number_of_tries+blockDimX-1)/blockDimX;
	unsigned int gridDimY=numberOfCorrectMoves;
	unsigned int numberOfThreads=blockDimX*gridDimX*gridDimY;

//	vector<pair<int,int> > moves_values;

	curandGenerator_t generator;
	randStatus=curandCreateGenerator(&generator,CURAND_RNG_PSEUDO_DEFAULT);
	check_rand_result(randStatus);
	randStatus=curandSetPseudoRandomGeneratorSeed(generator,123ULL);
	check_rand_result(randStatus);

	int size=Board::BOARD_SIZE*Board::BOARD_SIZE;

	//CUdeviceptr random_numbers;
	unsigned int*random_numbers;

	int const numberOfRandoms=number_of_tries*numberOfCorrectMoves*64;

	//randStatus=
	cudaMalloc((void**)&random_numbers,numberOfRandoms*sizeof(unsigned int));//???
	//check_rand_result(randStatus,"Cannot allocate array for randoms");

	randStatus=curandGenerate(generator,random_numbers,numberOfRandoms);
	check_rand_result(randStatus);

	unsigned int moveValues[64];

	/// calculate moves values using CUDA
	//for(auto move : correct_moves)
	//{
	//auto board_to_check=board.make_move(move);
	//int result=0;

	CUdeviceptr threadResults;
	result=cuMemAlloc(&threadResults,numberOfThreads*sizeof(int));
	check_result(result,"Results array malloc failed");

//	int tries=this->number_of_tries;
//	CUdeviceptr result_array_device;
//	if(cuMemAlloc(&result_array_device,this->number_of_tries*sizeof(int))!=CUDA_SUCCESS)
//	{
//		print_error("failed to alloc sth_changed\n");
//	}

	void*args[]={&board.player_pieces,&board.opponent_pieces,&threadResults};
	if(cuLaunchKernel(check_move,gridDimX,gridDimY,1,blockDimX,1,1,0,0,args,0)!=CUDA_SUCCESS)
		print_error("cannot run kernel check_move\n");

	if(cuCtxSynchronize()!=CUDA_SUCCESS)
		print_error("failed to synchronize\n");

	int*values=(int*)malloc(numberOfThreads*sizeof(int));
	if(cuMemcpyDtoH(values,threadResults,numberOfThreads*sizeof(int))!=CUDA_SUCCESS)
		print_error("failed to copy result from result_array_device");

	int bestMove=-1;
	int bestValue=INT_MIN;

	for(int i=0;i<numberOfCorrectMoves;i++)
	{
		long long sum=0;
		for(int j=0;j<number_of_tries;j++)
			sum+=values[gridDimX*blockDimX*i+j];
		if(sum>bestValue)
		{
			bestValue=sum;
			bestMove=correct_moves[i];
		}
	}

	return bestMove;
}


