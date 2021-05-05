from env import OthelloEnv
from copy import deepcopy
import numpy as np

class AlphaBeta():
    def __init__(self, env):
        self.env = env
        self.depth = 3


    def set_board(self, board, turn):
        self.env = OthelloEnv(board, turn)

    def pvs(self, alpha, beta, depth, board, color):
        if depth <= 0 or board.done:
            return self.evaluate(board) * color
        
        possible_moves = board.actions
        score = 0

        for index, x in enumerate(possible_moves):
            board.step(x)
            if index == 0:
                score = -self.pvs(-beta, -alpha, depth-1, deepcopy(board), -color)
                board.pop()
            else:
                score = -self.pvs(-alpha-1, -alpha, depth-1, deepcopy(board), -color)
                if score > alpha and score < beta:
                    score = -self.pvs(-beta, -alpha, depth-1, deepcopy(board), -color)
                board.pop()
            alpha = max(alpha, score)
            if alpha >= beta:
                break

        return score
    
    def evaluate(self, board):
        total = 0
        if board.done:
            return board.reward * 1000000

        weight_matrix = [
            'n', 'n', 150, 175, 175, 150, 'n', 'n',
            'n', 'n', 0, 0, 0, 0, 'n', 'n',
            150, 0, 15, 15, 15, 15, 0, 150,
            175, 0, 15, 25, 25, 15, 0, 175,
            175, 0, 15, 25, 25, 15, 0, 175,
            150, 0, 15, 15, 15, 15, 0, 150,
            'n', 'n', 0, 0, 0, 0, 'n', 'n',
            'n', 'n', 150, 175, 175, 150, 'n', 'n'
        ]

        corners = [0, 7, 56, 63]
        corner_dirs = [(1, 8, 9), (-1, 7, 8), (-8, -7, 1), (-9, -8, -1)]
        for i, corner in enumerate(corners):
            if board.state[corner] != 0:
                total += board.state[corner] * 1000
                for direction in corner_dirs[i]:
                    if board.state[corner + direction] == board.state[corner]:
                        total += board.state[corner] * 500
                    elif board.state[corner + direction] == -board.state[corner]:
                        total += board.state[corner + direction] * -1200
            else:
                for direction in corner_dirs[i]:
                    if board.state[corner + direction] != 0:
                        # print("DO NOT PLACE IN CORNER SQUARES")
                        total += board.state[corner + direction] * -900
        for index, x in enumerate(board.state):
            if weight_matrix[index] != 'n':
                total += x * weight_matrix[index]
        
        diff_tokens = board.state.sum()
        num_tokens = (board.state == 1).sum() + (board.state == -1).sum()
        mobility_weight = 100

        if num_tokens < 25:
            if diff_tokens < 0:
                total += abs(diff_tokens) * 30
            else:
                total -= abs(diff_tokens) * 30
        elif num_tokens < 53:
            mobility_weight = 25
            if diff_tokens < 0:
                total += abs(diff_tokens) * 15
            else:
                total -= abs(diff_tokens) * 15
        else:
            mobility_weight = 5
            if diff_tokens < 0:
                total -= abs(diff_tokens) * 125
            else:
                total += abs(diff_tokens) * 125

            
        mobility_black = board.actions_without_env(board.state, -1)
        mobility_white = board.actions_without_env(board.state, 1)

        if mobility_black == 64: mobility_black = [64]
        if mobility_white == 65: mobility_white = [65]

        total += (len(mobility_white) - len(mobility_black)) * mobility_weight

        return total
    
    def play(self):
        max_score = -99999999999999
        best_move = 0
        if self.env.actions == 64 or self.env.actions == 65:
            return self.env.actions

        for move in self.env.actions:
            self.env.step(move)
            score = -self.pvs(-100000000, 100000000, self.depth - 1, deepcopy(self.env), self.env.player_to_move)
            self.env.pop()

            if score > max_score:
                max_score = score
                best_move = move
        return best_move


if __name__ == '__main__':
    import random

    env = OthelloEnv()
    agent = AlphaBeta(deepcopy(env))
    env.render()

    random_player = OthelloEnv.BLACK
    ai_player = OthelloEnv.WHITE

    random_wins = 0
    ai_wins = 0

    for x in range(1):
        print(f'Game {x}')
        env.reset()
        done = False
        turn = -1

        while not done:
            if turn == ai_player:
                agent.set_board(np.copy(env.state), env.player_to_move)
                action = agent.play()
                observation, reward, done, turn = env.step(action)
            else:
                actions = env.actions
                if actions != 64 and actions != 65:
                    action = random.choice(env.actions)
                else:
                    action = actions
                observation, reward, done, turn = env.step(action)
            print('\n')
            env.render()
        if reward == ai_player:
            print('AI wins')
            ai_wins += 1
        elif reward == random_player:
            print('Random wins')
            random_wins += 1

    print(ai_wins, random_wins)