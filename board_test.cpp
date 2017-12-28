#include <iostream>
#include <cassert>
#include <algorithm>

#include "board.h"
#include "board_factory.h"
#include "board_exception.h"

using namespace std;

class BoardTests {
	private:
		BoardFactory board_factory = BoardFactory();
		Board starting_board = board_factory.get_starting_board();
	
	public:
		
	void test_get_bit_number() {
		assert(0 == Board::get_cell_number(0,0));
		assert(5 == Board::get_cell_number(0,5));
		assert(8 == Board::get_cell_number(1,0));
		assert(63 == Board::get_cell_number(7,7));
	}
	
	void test_get_site() {
		Board board=Board(0b1,0b1000);
		assert(Board::Site::PLAYER == board.get_site(0));
		assert(Board::Site::OPPONENT == board.get_site(3));
		assert(Board::Site::NONE == board.get_site(13));
	}
	
	void test_is_correct_move() {
		collection correct_moves = collection({make_pair(2,3), make_pair(3,2), make_pair(5,4),make_pair(4,5)});
		for(int i=0; i<8; i++)
			for(int j=0; j<8; j++) {
				auto cell=Board::get_cell_number(i,j);
				assert(	(find(correct_moves.begin(), correct_moves.end(), make_pair(i,j)) != correct_moves.end()) ==
						starting_board.is_correct_move(cell));
			}

		for(int i : list<int>({-1,8}))
			for(int j : list<int>({-1,8})) {
				auto cell = Board::get_cell_number(i,j);
				assert(starting_board.is_correct_move(cell) == false);
			}
	}
	
	void test_can_player_put_piece() {
		assert(starting_board.can_player_put_piece() == true);
		auto completed_board = Board(0b111,0);
		assert(completed_board.can_player_put_piece() == false);
	}
	
	void test_make_move() {
		auto expected = board_factory.get_board(collection({make_pair(4,4)}),
						collection({make_pair(2,3), make_pair(3,3), make_pair(4,3), make_pair(3,4)}));
		auto actual = starting_board.make_move(Board::get_cell_number(2,3));
		assert(expected == actual);
	}
	
	void test_pass_turn() {
		auto completed_board = Board(0b111,0b0);
		auto expected = Board(0b0,0b111);
		auto actual = completed_board.pass_turn();
		assert(expected == actual);
	}
	
	void test_pass_turn_fail() {
		try {
			starting_board.pass_turn();
		} catch (BoardException) {
			return;
		}
		assert(false);
	}
	
	void test_get_dominating_site() {
		auto player_more = Board(0b111,0b1000);
		auto opponent_more = Board(0b100,0b1011);
		auto equal_board = Board(0b111,0b111000);
		assert(Board::Site::PLAYER == player_more.get_dominating_site());
		assert(Board::Site::OPPONENT == opponent_more.get_dominating_site());
		assert(Board::Site::NONE == equal_board.get_dominating_site());
	}
	
	void test_text_representation() {
		string expected = string("  1 2 3 4 5 6 7 8 \n")+
				"  - - - - - - - - \n"+
				"1| | | | | | | | |\n"+
				"  - - - - - - - - \n"+
				"2| | | | | | | | |\n"+
				"  - - - - - - - - \n"+
				"3| | | |.| | | | |\n"+
				"  - - - - - - - - \n"+
				"4| | |.|O|X| | | |\n"+
				"  - - - - - - - - \n"+
				"5| | | |X|O|.| | |\n"+
				"  - - - - - - - - \n"+
				"6| | | | |.| | | |\n"+
				"  - - - - - - - - \n"+
				"7| | | | | | | | |\n"+
				"  - - - - - - - - \n"+
				"8| | | | | | | | |\n"+
				"  - - - - - - - - \n";
		auto actual = starting_board.text_representation();
		assert(expected == actual);
	}
	
	void test_incorrect_board() {
		try {
			Board(0b010,0b011);
		} catch (BoardException) {
			return;
		}
		assert(false);
	}
	
	void test_is_game_ended() {
		assert(starting_board.is_game_ended() == false);
		auto ended_game = Board(0b000,0b111);
		assert(ended_game.is_game_ended() == true);
	}
	
	void test_get_score() {
		auto board = Board(0b11010,0b00101);
		auto actual_player = board.get_score(Board::Site::PLAYER);
		auto actual_opponent = board.get_score(Board::Site::OPPONENT);
		auto expected_player = 3;
		auto expected_opponent = 2;
		assert(expected_player == actual_player);
		assert(expected_opponent == actual_opponent);
	}
	
	void test_get_move_value() {
		auto board = board_factory.get_board(
				collection({make_pair(2,4), make_pair(3,3), make_pair(3,4), make_pair(4,4)}),
				collection({make_pair(2,1), make_pair(2,2), make_pair(2,3), make_pair(4,1), make_pair(4,2), make_pair(4,3), make_pair(5,3)}));

		auto moves = collection({make_pair(1,2), make_pair(1,3), make_pair(2,0), make_pair(4,0), make_pair(5,2), make_pair(6,2), make_pair(6,3)});
		int results[7] = {1,1,3,3,1,1,2};

		int i=0;
		
		for(auto move : moves) {
			auto row = move.first;
			auto column = move.second;
			auto cell = Board::get_cell_number(row,column);
			assert(results[i] == board.get_move_value(cell));
			i++;
		}
	}
	
	void test_get_empty_fields() {
		auto expected = 60;
		auto actual = starting_board.get_empty_fields();
		assert(expected == actual);
	}

};

int main() {
	BoardTests board_tests;
	
	board_tests.test_get_bit_number();
	board_tests.test_get_site();
	board_tests.test_is_correct_move();
	board_tests.test_can_player_put_piece();
	board_tests.test_make_move();
	board_tests.test_pass_turn();
	board_tests.test_pass_turn_fail();
	board_tests.test_get_dominating_site();
	board_tests.test_text_representation();
	board_tests.test_incorrect_board();
	board_tests.test_is_game_ended();
	board_tests.test_get_score();
	board_tests.test_get_move_value();
	board_tests.test_get_empty_fields();
}
