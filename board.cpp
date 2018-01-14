#include "board.h"
#include "board_exception.h"

#ifdef __CUDACC__
#define CUDA_CALLABLE_MEMBER __device__
#else
#define CUDA_CALLABLE_MEMBER
#endif

/** public **/
const int Board::BOARD_SIZE=8;
/*
static const cell_t Board::DIRECTIONS[]={cell_t(-1,-1), //NW
								  cell_t(-1,0), //N
								  cell_t(-1,1), //NE
								  cell_t(0,-1), //W
								  cell_t(0,1), //E
								  cell_t(1,-1), //SW
								  cell_t(1,0), //S
								  cell_t(1,1) //SE
};
*/
CUDA_CALLABLE_MEMBER
Board::Board(ull player_pieces,ull opponent_pieces)
{
	this->player_pieces=player_pieces;
	this->opponent_pieces=opponent_pieces;
	if(!(this->is_correct_board()))
	{
#ifndef __CUDACC__
		throw BoardException();
#endif
	}
}

CUDA_CALLABLE_MEMBER
int Board::get_cell_number(int row,int column)
{
	return row*Board::BOARD_SIZE+column;
}

CUDA_CALLABLE_MEMBER
cell_t Board::get_cell_coordinates(int cell)
{
	return cell_t{cell/8,cell%8};
}

CUDA_CALLABLE_MEMBER
bool Board::is_correct_coordinate(int coordinate)
{
	return (coordinate<8&&coordinate>=0);
}

CUDA_CALLABLE_MEMBER
int Board::get_empty_fields()
{
	return Board::BOARD_SIZE*Board::BOARD_SIZE-bit_count(this->player_pieces)-bit_count(this->opponent_pieces);
}

CUDA_CALLABLE_MEMBER
Board::Site Board::get_site(int cell)
{
	auto player_bit=get_nth_bit(this->player_pieces,cell);
	auto opponent_bit=get_nth_bit(this->opponent_pieces,cell);
	if(player_bit==1)
		return Board::Site::PLAYER;
	if(opponent_bit==1)
		return Board::Site::OPPONENT;
	return Board::Site::NONE;
}

CUDA_CALLABLE_MEMBER
Board Board::pass_turn()
{
	if(can_player_put_piece())
	{
#ifndef __CUDACC__
		throw BoardException();
#endif
	}
	return Board(this->opponent_pieces,this->player_pieces);
}

CUDA_CALLABLE_MEMBER
bool Board::is_game_ended()
{
	if(can_player_put_piece())
		return false;
	if(pass_turn().can_player_put_piece())
		return false;
	return true;
}

CUDA_CALLABLE_MEMBER
Board Board::make_move(int cell)
{
	auto cell_pair=get_cell_coordinates(cell);
	auto row=cell_pair.first;
	auto column=cell_pair.second;

	auto flip_mask=0ULL;
	for(auto direction : {cell_t(-1,-1), //NW
						  cell_t(-1,0), //N
						  cell_t(-1,1), //NE
						  cell_t(0,-1), //W
						  cell_t(0,1), //E
						  cell_t(1,-1), //SW
						  cell_t(1,0), //S
						  cell_t(1,1) //SE
	})
	{
		auto direction_row=direction.first;
		auto direction_column=direction.second;

		auto current_row=row+direction_row;
		auto current_column=column+direction_column;
		if(!is_correct_coordinate(current_row)||!is_correct_coordinate(current_column))
			continue;
		auto current_cell=get_cell_number(current_row,current_column);
		if(get_site(current_cell)!=Board::Site::OPPONENT)
			continue;
		if(is_players_piece_on_the_end(current_cell,direction))
		{
			flip_mask=flip_mask|get_flip_mask(current_cell,direction);
		}
	}
	auto new_player_pieces=opponent_pieces^flip_mask;
	auto new_opponent_pieces=(player_pieces^flip_mask)|(1ULL<<cell);
	return Board(new_player_pieces,new_opponent_pieces);
}

CUDA_CALLABLE_MEMBER
bool Board::is_correct_move(int cell)
{
	auto cell_pair=get_cell_coordinates(cell);
	auto row=cell_pair.first;
	auto column=cell_pair.second;

	if(!is_correct_coordinate(row)||!is_correct_coordinate(column))
		return false;
	if(get_site(cell)!=Board::Site::NONE)
		return false;
	for(auto direction : {cell_t(-1,-1), //NW
						  cell_t(-1,0), //N
						  cell_t(-1,1), //NE
						  cell_t(0,-1), //W
						  cell_t(0,1), //E
						  cell_t(1,-1), //SW
						  cell_t(1,0), //S
						  cell_t(1,1) //SE
	})
	{
		auto direction_row=direction.first;
		auto direction_column=direction.second;

		auto current_row=row+direction_row;
		auto current_column=column+direction_column;

		if(!is_correct_coordinate(current_row)||!is_correct_coordinate(current_column))
			continue;
		auto current_cell=get_cell_number(current_row,current_column);
		if(get_site(current_cell)!=Board::Site::OPPONENT)
			continue;
		if(is_players_piece_on_the_end(current_cell,direction))
			return true;
	}
	return false;
}

CUDA_CALLABLE_MEMBER
bool Board::can_player_put_piece()
{
	for(int i=0;i<Board::BOARD_SIZE*Board::BOARD_SIZE;i++)
	{
		if(is_correct_move(i)) return true;
	}
	return false;
}

#ifndef __CUDACC__
string Board::text_representation(char player_sign,char opponent_sign)
{

	string row_separator="  - - - - - - - - \n";
	string result="  1 2 3 4 5 6 7 8 \n";
	result+=row_separator;
	for(int i=0;i<8;i++)
	{
		result+=to_string(i+1);
		for(int j=0;j<8;j++)
		{
			result+='|';
			auto cell=get_cell_number(i,j);
			auto site=get_site(cell);
			string to_append;
			switch(site)
			{
				case Board::Site::PLAYER:
					to_append=player_sign;
					break;
				case Board::Site::OPPONENT:
					to_append=opponent_sign;
					break;
				case Board::Site::NONE:
					to_append=is_correct_move(cell)?'.':' ';
			}
			result+=to_append;
		}
		result+="|\n";
		result+=row_separator;
	}

	return result;

}
#endif
CUDA_CALLABLE_MEMBER
Board::Site Board::get_dominating_site()
{
	if(bit_count(this->player_pieces)>bit_count(this->opponent_pieces))
		return Board::Site::PLAYER;
	if(bit_count(this->player_pieces)<bit_count(this->opponent_pieces))
		return Board::Site::OPPONENT;
	return Board::Site::NONE;
}

CUDA_CALLABLE_MEMBER
int Board::get_score(Board::Site site)
{
	return (site==Board::Site::PLAYER)?bit_count(this->player_pieces):bit_count(this->opponent_pieces);
}

CUDA_CALLABLE_MEMBER
int Board::get_move_value(int move)
{
	if(!is_correct_move(move))
	{
#ifndef __CUDACC__
		throw BoardException();
#endif
	}
	auto board_after_move=make_move(move);
	return board_after_move.get_score(Board::Site::OPPONENT)-get_score(Board::Site::PLAYER)-1;
}


/** private **/
CUDA_CALLABLE_MEMBER
int Board::bit_count(ull value)
{
	auto result=0;
	while(value!=0ULL)
	{
		result+=(int)(value&1);
		value=value>>1;
	}
	return result;
}

CUDA_CALLABLE_MEMBER
int Board::get_nth_bit(ull value,int n)
{
	return (int)((value>>n)&1ULL);
}

CUDA_CALLABLE_MEMBER
bool Board::is_correct_board()
{
	return ((this->player_pieces&this->opponent_pieces)==0ULL);
}

CUDA_CALLABLE_MEMBER
ull Board::get_flip_mask(int cell,cell_t direction)
{
	auto result=0ULL;
	auto current_cell=cell;
	auto direction_row=direction.first;
	auto direction_column=direction.second;

	while(get_site(current_cell)==Board::Site::OPPONENT)
	{
		result=result|(1ULL<<current_cell);
		auto cell_pair=get_cell_coordinates(current_cell);
		auto row=cell_pair.first;
		auto column=cell_pair.second;
		auto current_row=row+direction_row;
		auto current_column=column+direction_column;
		current_cell=get_cell_number(current_row,current_column);
	}
	return result;
}

CUDA_CALLABLE_MEMBER
bool Board::is_players_piece_on_the_end(int cell,cell_t direction)
{
	auto current_cell=cell;
	auto direction_row=direction.first;
	auto direction_column=direction.second;

	while(get_site(current_cell)==Board::Site::OPPONENT)
	{
		auto cell_pair=get_cell_coordinates(current_cell);
		auto row=cell_pair.first;
		auto column=cell_pair.second;
		auto current_row=row+direction_row;
		auto current_column=column+direction_column;

		if(!is_correct_coordinate(current_row)||!is_correct_coordinate(current_column))
			return false;
		current_cell=get_cell_number(current_row,current_column);
	}
	return get_site(current_cell)==Board::Site::PLAYER;
}	

