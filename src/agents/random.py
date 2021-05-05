import random
import copy

class RandomOthelloRunner:

    def __init__(self):
        self.white = "o"
        self.black = "@"
        self.directions = [[-1, -1], [-1, 0], [-1, 1], [0, -1], [0, 1], [1, -1], [1, 0], [1, 1]]
        self.opposite_color = {self.black: self.white, self.white: self.black}
        self.x_max = 8
        self.y_max = 8

    def best_strategy(self, board, color):
        # returns best move
        options = self.find_moves(board, color)
        best_move = random.choice(list(options.keys()))
        return [best_move // self.x_max, best_move % self.y_max], 0

    def find_moves(self, my_board, my_color):
        # finds all possible moves
        moves_found = {}
        for i in range(len(my_board)):
            for j in range(len(my_board[i])):
                flipped_stones = self.find_flipped(my_board, i, j, my_color)
                if len(flipped_stones) > 0:
                    moves_found.update({i * self.y_max + j: flipped_stones})
        return moves_found

    def find_flipped(self, my_board, x, y, my_color):
        # finds which chips would be flipped given a move and color
        if my_board[x][y] != ".":
            return []
        if my_color == '#000000':
            my_color = "@"
        else:
            my_color = "O"
        flipped_stones = []
        for incr in self.directions:
            temp_flip = []
            x_pos = x + incr[0]
            y_pos = y + incr[1]
            while 0 <= x_pos < self.x_max and 0 <= y_pos < self.y_max:
                if my_board[x_pos][y_pos] == ".":
                    break
                if my_board[x_pos][y_pos] == my_color:
                    flipped_stones += temp_flip
                    break
                temp_flip.append([x_pos, y_pos])
                x_pos += incr[0]
                y_pos += incr[1]
        return flipped_stones