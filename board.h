#ifndef BOARD
#define BOARD

#include <iostream>
#include <list>
#include "cuda.h"

#ifdef __CUDACC__
#define CUDA_CALLABLE_MEMBER __device__
#else
#define CUDA_CALLABLE_MEMBER
#endif 

using namespace std;
typedef unsigned long long ull;
typedef list<pair<int,int> > collection;

class Board {
	public:
		ull player_pieces;
		ull opponent_pieces;
		
		enum Site { PLAYER, OPPONENT, NONE };
		static const int BOARD_SIZE;
		static const collection DIRECTIONS;
		CUDA_CALLABLE_MEMBER
		static int get_cell_number(int row, int column);
		CUDA_CALLABLE_MEMBER
		static pair<int,int> get_cell_coordinates(int cell);
		CUDA_CALLABLE_MEMBER
		static bool is_correct_coordinate(int coordinate);		
		
		CUDA_CALLABLE_MEMBER
		Board(ull player_pieces, ull opponent_pieces);
		CUDA_CALLABLE_MEMBER
		int get_empty_fields();
		CUDA_CALLABLE_MEMBER
		Site get_site(int cell);
		CUDA_CALLABLE_MEMBER
		Board pass_turn();
		CUDA_CALLABLE_MEMBER
		bool is_game_ended();
		CUDA_CALLABLE_MEMBER
		Board make_move(int cell);
		CUDA_CALLABLE_MEMBER
		bool is_correct_move(int cell);
		CUDA_CALLABLE_MEMBER
		bool can_player_put_piece();
		CUDA_CALLABLE_MEMBER
		string text_representation(char player_sign='X', char opponent_sign='O');
		CUDA_CALLABLE_MEMBER
		Site get_dominating_site();
		CUDA_CALLABLE_MEMBER
		int get_score(Site site); 
		CUDA_CALLABLE_MEMBER
		int get_move_value(int move);
		
	private:
		CUDA_CALLABLE_MEMBER
		static int bit_count(ull value);
		CUDA_CALLABLE_MEMBER
        int get_nth_bit(ull value, int n);
        CUDA_CALLABLE_MEMBER
		bool is_correct_board();
		CUDA_CALLABLE_MEMBER
		ull get_flip_mask(int cell, pair<int,int> direction);  
		CUDA_CALLABLE_MEMBER
		bool is_players_piece_on_the_end(int cell, pair<int,int> direction);
};

CUDA_CALLABLE_MEMBER
inline bool operator==(const Board& lhs, const Board& rhs) { 
	return (lhs.player_pieces == rhs.player_pieces) && (lhs.opponent_pieces == rhs.opponent_pieces); 
}
	

#endif
