#ifndef BOARD
#define BOARD

#include <iostream>
#include <list>

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
		static int get_cell_number(int row, int column);
		static pair<int,int> get_cell_coordinates(int cell);
		static bool is_correct_coordinate(int coordinate);		
		
		Board(ull player_pieces, ull opponent_pieces);
		int get_empty_fields();
		Site get_site(int cell);
		Board pass_turn();
		bool is_game_ended();
		Board make_move(int cell);
		bool is_correct_move(int cell);
		bool can_player_put_piece();
		string text_representation(char player_sign='X', char opponent_sign='O');
		Site get_dominating_site();
		int get_score(Site site); 
		int get_move_value(int move);
		
	private:
		static int bit_count(ull value);
        int get_nth_bit(ull value, int n);
		bool is_correct_board();
		ull get_flip_mask(int cell, pair<int,int> direction);  
		bool is_players_piece_on_the_end(int cell, pair<int,int> direction);
};

inline bool operator==(const Board& lhs, const Board& rhs) { 
	return (lhs.player_pieces == rhs.player_pieces) && (lhs.opponent_pieces == rhs.opponent_pieces); 
}
	

#endif
