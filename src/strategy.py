import math

EMPTY, BLACK, WHITE, EDGE = '.', '@', 'o', '?'
DIRECTIONS = (-11, -10, -9, -1, 1, 9, 10, 11)
OPPONENT = {BLACK: WHITE, WHITE: BLACK}
CORNERS = (11, 18, 81, 88)
CORNER_ADJ = (12, 21, 22, 17, 27, 28, 71, 72, 82, 87, 77, 78)
ADJ_TO_CORNER = {
    12: 11,
    21: 11,
    22: 11,
    17: 18,
    27: 18,
    28: 18,
    71: 81,
    72: 81,
    82: 81,
    87: 88,
    77: 88,
    78: 88
}


class Strategy:
    def __init__(self):
        self.explored = {}

    def best_strategy(self, board, player, best_move, still_running):
        depth = 1
        while still_running.value:
            best_move.value = self.alphabeta(board, player, depth, -math.inf, math.inf)
            depth += 1

    def find_moves(self, my_board, my_color):
        moves_found = {}
        for i in range(len(my_board)):
            if my_board[i] == EMPTY:
                flipped_stones = self.find_flipped(my_board, i, my_color)
                if len(flipped_stones) > 0:
                    moves_found.update({i: flipped_stones})
        return moves_found

    def find_flipped(self, my_board, index, my_color):
        flipped_stones = []
        for incr in DIRECTIONS:
            temp_flip = []
            curr_index = index + incr
            while my_board[curr_index] != EDGE:
                if my_board[curr_index] == EMPTY:
                    break
                if my_board[curr_index] == my_color:
                    flipped_stones += temp_flip
                    break
                temp_flip.append([curr_index])
                curr_index += incr
        return flipped_stones

    def alphabeta(self, board, color, depth, alpha, beta):
        return self.max_value(board, color, depth, alpha, beta)[1] if color == BLACK else \
        self.min_value(board, color, depth, alpha, beta)[1]

    def terminal_test(self, board):
        if not board.find(EMPTY): return True
        black = board.find(BLACK)
        if not black: return True
        white = board.find(WHITE)
        if not white: return True
        black_moves = len(self.find_moves(board, BLACK))
        white_moves = len(self.find_moves(board, WHITE))
        if not black_moves and not white_moves: return True
        return False

    def max_value(self, board, color, depth, alpha, beta):
        if depth == 0 or self.terminal_test(board):
            if board not in self.explored:
                self.explored[board] = self.evaluate(board)
            return self.explored[board], 0

        v = -math.inf
        possible_moves = self.find_moves(board, color)
        if len(possible_moves.keys()) <= 0: return self.min_value(board, OPPONENT[color], depth - 1, alpha, beta)
        choice = list(possible_moves.keys())[0]

        for move in possible_moves:
            # greedy corner grab
            if move in CORNERS:
                if board not in self.explored:
                    self.explored[board] = self.evaluate(board)
                return self.explored[board], move

            eval = self.min_value(self.make_move(board, color, move, possible_moves[move]),
                                  OPPONENT[color], depth - 1, alpha, beta)
            if eval[0] > v:
                v = eval[0]
                choice = move
            if v >= beta: return v, choice
            alpha = max(alpha, v)
        return v, choice

    def min_value(self, board, color, depth, alpha, beta):
        if depth == 0 or self.terminal_test(board):
            if board not in self.explored:
                self.explored[board] = self.evaluate(board)
            return self.explored[board], 0

        v = math.inf
        possible_moves = self.find_moves(board, color)
        if len(possible_moves.keys()) <= 0: return self.max_value(board, OPPONENT[color], depth - 1, alpha, beta)
        choice = list(possible_moves.keys())[0]

        for move in possible_moves:
            # greedy corner grab
            if move in CORNERS:
                if board not in self.explored:
                    self.explored[board] = self.evaluate(board)
                return self.explored[board], move

            eval = self.max_value(self.make_move(board, color, move, possible_moves[move]),
                                  OPPONENT[color], depth - 1, alpha, beta)
            if eval[0] < v:
                v = eval[0]
                choice = move
            if v <= alpha: return v, choice
            beta = min(beta, v)
        return v, choice

    def make_move(self, board, color, move, flipped):
        # returns board that has been updated
        copy_board = board
        copy_board = copy_board[:move] + color + copy_board[move + 1:]
        for f in flipped:
            copy_board = copy_board[:move] + color + copy_board[move + 1:]
        return copy_board

    def evaluate(self, board):
        evaluation = [0, 0, 0, 0]
        # ----------mobility----------#
        black_move = self.find_moves(board, BLACK)
        white_move = self.find_moves(board, WHITE)
        if len(black_move) + len(white_move) != 0:
            evaluation[0] = 100 * ((len(black_move) - len(white_move)) / (len(black_move) + len(white_move)))
        else:
            evaluation[0] = 0

            # ----------number of stones----------#
        black = board.count(BLACK)
        white = board.count(WHITE)

        if black == 0: return -math.inf  # instant win for white
        if white == 0: return math.inf  # instant win for black
        if black + white >= 64:  # game over
            if black > white: return math.inf  # instant win for black
            if black < white: return -math.inf  # instant win for white

        if black + white != 0:
            if black + white > 42:
                evaluation[1] = 100 * ((black - white) / (black + white))
            else:
                evaluation[1] = 25 * ((black - white) / (black + white))
        else:
            evaluation[1] = 0

            # ----------corners----------#
        black_corner, white_corner = 0, 0
        for corner in CORNERS:
            if board[corner] == BLACK:
                black_corner += 1
            elif board[corner] == WHITE:
                white_corner += 1
        if black_corner + white_corner != 0:
            evaluation[2] = 250 * ((black_corner - white_corner) / (black_corner + white_corner))
        else:
            evaluation[2] = 0

        # ----------corner potential----------#
        black_cp, white_cp = 0, 0
        for corner in CORNERS:
            if corner in black_move:
                black_cp += 1
            if corner in white_move:
                white_cp += 1

        for adj in CORNER_ADJ:
            if board[adj] == BLACK and board[ADJ_TO_CORNER[adj]] != BLACK:
                white_cp += 1
            if board[adj] == WHITE and board[ADJ_TO_CORNER[adj]] != WHITE:
                black_cp += 1

        if black_cp + white_cp != 0:
            evaluation[3] = 300 * ((black_cp - white_cp) / (black_cp + white_cp))
        else:
            evaluation[3] = 0

        return sum(evaluation)
