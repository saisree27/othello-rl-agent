import random
import copy

class MinimaxOthelloRunner:

    def __init__(self):
        self.white = "o"
        self.black = "@"
        self.directions = [[-1, -1], [-1, 0], [-1, 1], [0, -1], [0, 1], [1, -1], [1, 0], [1, 1]]
        self.opposite_color = {self.black: self.white, self.white: self.black}
        self.x_max = 8
        self.y_max = 8

    def best_strategy(self, board, color):
        # returns best move
        best_move = self.minimax(board, color, 3)
        return [best_move // self.x_max, best_move % self.y_max], 0

    def minimax(self, board, color, search_depth):
        # returns best "value"
        return self.max_value(board, color, search_depth)[1]

    def terminal_test(self, board, color):
        return len(self.find_moves(board, color)) == 0

    def max_value(self, board, color, depth):
        # return value and statef
        if self.terminal_test(board, color) or depth == 0:
            return self.evaluate(board, color, self.find_moves(board, color)), board
        next_color = '#ffffff' if color == '#000000' else '#000000'
        v = -9999
        choice = ()
        successors = self.find_moves(board, color)
        for child in successors:
            eval = self.min_value(self.make_move(board, color, child, successors[child]), next_color, depth - 1)
            if eval[0] > v:
                v = eval[0]
                choice = child
        return v, choice

    def min_value(self, board, color, depth):
        # return value and state
        if self.terminal_test(board, color) or depth == 0:
            return -1 * self.evaluate(board, color, self.find_moves(board, color)), board
        next_color = '#ffffff' if color == '#000000' else '#000000'
        v = 9999
        choice = ()
        successors = self.find_moves(board, color)
        for child in successors:
            eval = self.max_value(self.make_move(board, color, child, successors[child]), next_color, depth - 1)
            if eval[0] < v:
                v = eval[0]
                choice = child
        return v, choice

    def negamax(self, board, color, search_depth):
        # returns best "value"
        return 1

    def alphabeta(self, board, color, search_depth, alpha, beta):
        # returns best "value" while also pruning
        pass

    def make_key(self, board, color):
        # hashes the board
        return 1

    def stones_left(self, board):
        # returns number of stones that can still be placed
        return 1

    def make_move(self, board, color, move, flipped):
        # returns board that has been updated
        copy_board = copy.deepcopy(board)
        copy_board[move // self.x_max][move % self.y_max] = '@' if color == '#000000' else 'O'
        for f in flipped:
            copy_board[f[0]][f[1]] = '@' if color == '#000000' else 'O'
        return copy_board

    def evaluate(self, board, color, possible_moves):
        # returns the utility value
        if len(possible_moves) == 0: return -200
        opposite_turn = '#000000' if color == '#ffffff' else '#ffffff'
        opposition = self.find_moves(board, opposite_turn)
        if len(opposition) == 0:
            return 200
        return len(possible_moves) - len(opposition)

    def score(self, board, color):
        # returns the score of the board
        return 1

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
