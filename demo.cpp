#include <iostream>

#include "board.h"
#include "randomized_play_player.h"

using namespace std;

int main() {

	int number_of_tries=100*1024;

	RandomizedPlayPlayerCuda randomized_play_player_cuda("Player", 12345, number_of_tries);
	
	BoardFactory board_factory = BoardFactory();
	Board starting_board = board_factory.get_starting_board();
	cerr << starting_board.text_representation('X','O');
	
	auto move = randomized_play_player_cuda.make_move(starting_board);
	Board new_board = starting_board = starting_board.make_move(move);
	cerr << new_board.text_representation('O','X');
}

