#ifndef RANDOMIZED_PLAY_PLAYER
#define RANDOMIZED_PLAY_PLAYER

#include "board.h"
#include "board_factory.h"
#include "board_exception.h"

using namespace std;

class RandomizedPlayPlayerCuda {
	public:
		string name;		
		RandomizedPlayPlayerCuda(string name, int seed, int number_of_tries);
		int make_move(Board board);
	private:
		int seed;
		int number_of_tries;
};

#endif
