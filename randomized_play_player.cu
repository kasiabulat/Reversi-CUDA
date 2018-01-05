#include "cuda.h"
#include "curand.h"

#include <cstdio>
#include <stdlib.h> 

#include "board.h"
#include "board_factory.h"

extern "C" {
	__device__
	int evaluate(Board board, unsigned int * random_number) {
		if(board.is_game_ended()) {
			switch(board.get_dominating_site()) {
				case Board::PLAYER:
					return 1;
				case Board::OPPONENT:
					return -1;
                case Board::NONE:
					return 0;
			}
		}
		if(!board.can_player_put_piece()) {
			return -evaluate(board.pass_turn(), random_number);
		}
		
		int correct_moves[64];
		int idx = 0;
		for(int i=0; i<64; i++) 
			if(board.is_correct_move(i)) 
				correct_moves[idx++] = i;
		
		int element_number = *(random_number++); 	
	
		int played_move = correct_moves[element_number];
		Board next_board = board.make_move(played_move);
		return -evaluate(next_board, random_number);
	}
	
	__global__
	void check_move(int seed, ull player_pieces, ull opponent_pieces, int * result) {
		int thid = (blockIdx.x * blockDim.x) + threadIdx.x;
				
	
		// todo: make board from player_pieces and opponent_pieces	
		//result[thid] = evaluate(board, random_numbers);	
	}
}


