#include <iostream>
#include <list>
#include "board.h"

using namespace std;
typedef long long ll;
typedef list<pair<int,int> > collection;

class BoardFactory {
	public:
		Board get_board(collection player_collection, collection opponent_collection);
		Board get_starting_board();
	private:
		ll get_state(collection cells);
};
