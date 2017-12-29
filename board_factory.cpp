#include "board.h"
#include "board_factory.h"

Board BoardFactory::get_board(collection player_collection, collection opponent_collection) {
	auto player_pieces = get_state(player_collection);
	auto opponent_pieces = get_state(opponent_collection);				
	return Board(player_pieces,opponent_pieces);
}

Board BoardFactory::get_starting_board() {
	return get_board(collection({make_pair(4,3),make_pair(3,4)}), 
					 collection({make_pair(3,3),make_pair(4,4)}));
}

ull BoardFactory::get_state(collection cells) {
	ull result = 0ULL;
	for(auto cell : cells) 
		result |= (1ULL << Board::get_cell_number(cell.first, cell.second));		
	return result;
}
