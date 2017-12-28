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

ll BoardFactory::get_state(collection cells) {
	ll result = 0L;
	for(auto cell : cells) 
		result = result || (1L << Board::get_cell_number(cell.first, cell.second));		
	return result;
}
