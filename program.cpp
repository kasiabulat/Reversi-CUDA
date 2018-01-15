#include <iostream>
#include "board.h"
#include "randomized_play_player.h"


using namespace std;

int main(int argc,char**argv)
{
	if(argc!=3)
	{
		cerr<<"Wrong number of arguments."<<endl;
		exit(1);
	}
	int number_of_tries=100*1024;

	RandomizedPlayPlayerCuda randomized_play_player_cuda("Kasia", 12345, number_of_tries);

	unsigned long long playerPieces=stoull(string(argv[1]));
	unsigned long long opponentPieces=stoull(string(argv[2]));

	Board board=Board(playerPieces,opponentPieces);
	cerr<<board.text_representation()<<endl;
	auto move = randomized_play_player_cuda.make_move(board);
	cerr<<board.make_move(move).text_representation()<<endl;
	cout<<move<<endl;
}