#include <cuda.h>
#include <curand.h>
#include <cstdio>
#include <iostream>
#include <cstring>
#include <cassert>
#include <stdlib.h>
#include <vector>
#include <algorithm>
#include <cstring>

#include "board.h"
#include "randomized_play_player.h"

using namespace std;

void print_error(string error)
{
	cout<<error;
	exit(1);
}

RandomizedPlayPlayerCuda::RandomizedPlayPlayerCuda(string name,unsigned long long seed,int number_of_tries)
{
	this->name=name;
	this->seed=seed;
	this->number_of_tries=number_of_tries;

	cuInit(0);

	CUdevice cuDevice;
	CUresult res=cuDeviceGet(&cuDevice,0);
	if(res!=CUDA_SUCCESS)
	{
		printf("cannot acquire device 0\n");
		exit(1);
	}

	CUcontext cuContext;
	res=cuCtxCreate(&cuContext,0,cuDevice);
	if(res!=CUDA_SUCCESS)
	{
		printf("cannot create context\n");
		exit(1);
	}
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
		char const*errorName;
		cuGetErrorName(result,&errorName);
		char const*errorDescription;
		cuGetErrorString(result,&errorDescription);
		string error=message+": "+string(errorName)+" "+string(errorDescription)+"\n";

		print_error(error);
		exit(1);
	}
}

int RandomizedPlayPlayerCuda::make_move(Board board)
{
	curandStatus_t randStatus;
	CUresult result;

	/// load module
	CUmodule cuModuleRandomizedPlayPlayer;
	result=cuModuleLoad(&cuModuleRandomizedPlayPlayer,"randomized_play_player.ptx");
	check_result(result,"cannot load module");

	/// kernel functions
	CUfunction check_move;
	if(cuModuleGetFunction(&check_move,cuModuleRandomizedPlayPlayer,"check_move")!=CUDA_SUCCESS)
		print_error("cannot acquire kernel handle for check_move\n");

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

	curandGenerator_t generator;
	randStatus=curandCreateGenerator(&generator,CURAND_RNG_PSEUDO_DEFAULT);
	check_rand_result(randStatus);
	randStatus=curandSetPseudoRandomGeneratorSeed(generator,seed);
	check_rand_result(randStatus);

	int size=Board::BOARD_SIZE*Board::BOARD_SIZE;

	//CUdeviceptr random_numbers;
	unsigned int*random_numbers;

	int const numberOfRandoms=numberOfThreads*128;

	cudaMalloc((void**)&random_numbers,numberOfRandoms*sizeof(unsigned int));
	//check_rand_result(randStatus,"Cannot allocate array for randoms");

	randStatus=curandGenerate(generator,random_numbers,numberOfRandoms);
	check_rand_result(randStatus);

	/// calculate moves values using CUDA

	CUdeviceptr threadResults;
	result=cuMemAlloc(&threadResults,numberOfThreads*sizeof(int));
	check_result(result,"Results array malloc failed");

	CUdeviceptr correctMovesDevice;
	result=cuMemAlloc(&correctMovesDevice,numberOfCorrectMoves*sizeof(int));
	check_result(result,"Moves array malloc failed");

	result=cuMemcpyHtoD(correctMovesDevice,correct_moves,numberOfCorrectMoves*sizeof(int));
	check_result(result,"Moves array malloc failed");


	void*args[]={&board.player_pieces,&board.opponent_pieces,&correctMovesDevice,&random_numbers,&threadResults};
	result=cuLaunchKernel(check_move,gridDimX,gridDimY,1,blockDimX,1,1,0,0,args,0);
	check_result(result,"cannot run kernel check_move\n");

	result=cuCtxSynchronize();
	check_result(result,"failed to synchronize\n");

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
		cerr<<"Index: "<<i<<",sum: "<<sum<<endl;
		if(sum>bestValue)
		{
			bestValue=sum;
			bestMove=correct_moves[i];
		}
	}

	return bestMove;
}


