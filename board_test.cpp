#include <iostream>
#include "board.h"
#include "board_factory.h"

using namespace std;

class BoardTests {
	private:
		BoardFactory board_factory = BoardFactory();
		Board starting_board = board_factory.get_starting_board();
};

int main() {
	
}
