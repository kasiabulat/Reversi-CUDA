#ifndef BOARD_FACTORY
#define BOARD_FACTORY

#include <iostream>
#include <list>
#include "board.h"

using namespace std;
typedef unsigned long long ull;
typedef list<pair<int,int> > collection;

class BoardFactory {
	public:
		Board get_board(collection player_collection, collection opponent_collection);
		Board get_starting_board();
	private:
		ull get_state(collection cells);
};

#endif
