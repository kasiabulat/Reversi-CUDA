#include "cuda.h"
#include <cstdio>
#include <stdlib.h> 
#include "board.h"
#include "board_factory.h"

extern "C" {
	__device__
	int evaluate(Board board) {
		if(board.is_game_ended()) {
			switch(board.get_dominating_site()) {
				case Board::Site::PLAYER:
					return 1;
				case Board::Site::OPPONENT:
					return -1;
				case Board::Site::NONE:
					return 0;
			}
		}
		if(!board.can_player_put_piece()) {
			return -board.pass_turn().evaluate();
		}
		
		vector<int> correct_moves;
		for(int i=0; i<64; i++) 
			if(board.is_correct_move(i) 
				correct_moves.push_back(i);
		
		auto element_number = rand () % moves.size;	
		auto played_move = correct_moves[element_number];
		auto next_board = make_move(played_move)
		return -next_board.evaluate();
	}
	
	__global__
	void check_move(int seed, ull player_pieces, ull opponent_pieces, int * result) {
		srand(seed); // ?
		int thid = (blockIdx.x * blockDim.x) + threadIdx.x;
		// todo: make board from player_pieces and opponent_pieces	
		//result[thid] = evaluate(board);	
	}
}


