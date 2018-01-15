#include <cuda.h>
#include <curand.h>

#include <cstdio>
#include <stdlib.h>

#include "board.h"
#include "board_factory.h"
#include "board.cpp"

extern "C" {



__device__ int evaluate(Board board,unsigned int*random_numbers,unsigned int numberOfThreads,unsigned int globalId)
{
	int resultMultiplier=1;
	unsigned int correct_moves[64];
	for(unsigned int iteration=0;!board.is_game_ended();iteration++)
	{
		if(!board.can_player_put_piece())
		{
			board=board.pass_turn();
		}
		else
		{
			unsigned int idx=0;
			for(int i=0;i<64;i++)
				if(board.is_correct_move(i))
					correct_moves[idx++]=i;
			unsigned int element_number=random_numbers[numberOfThreads*iteration+globalId]%idx;
			board=board.make_move(correct_moves[element_number]);
		}
		resultMultiplier*=-1;

	}
	switch(board.get_dominating_site())
	{
		case Board::PLAYER:
			return 1*resultMultiplier;
		case Board::OPPONENT:
			return -1*resultMultiplier;
		case Board::NONE:
			return 0*resultMultiplier;
	}
	return -1;
}

__global__ void check_move(ull player_pieces,ull opponent_pieces,int*moves_to_check,unsigned int*randoms,int*result)
{
	int moveId=blockIdx.y;
	int thidX=(blockIdx.x*blockDim.x)+threadIdx.x;
	int numberOfTries=blockDim.x*gridDim.x;
	int threadGlobalId=numberOfTries*blockIdx.y+thidX;
	Board computedBoard(player_pieces,opponent_pieces);
	int checkedMove=moves_to_check[moveId];
	Board movedBoard=computedBoard.make_move(checkedMove);
	result[threadGlobalId]=-evaluate(movedBoard,randoms,numberOfTries,threadGlobalId);
}
}


