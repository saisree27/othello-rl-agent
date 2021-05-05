import gym
import numpy as np

class OthelloEnv():
    
    BLACK = -1
    WHITE = 1
    EMPTY = 0

    TOKENS = {BLACK: "@", WHITE:"o"}
    MAP = {"black": BLACK, "white": WHITE}

    MOVE_DIRS = {'U': -8, 'D': 8, 'L': -1, 'R': 1, 'UL': -9, 'UR': -7, 'DL': 7, 'DR': 9}

    def __init__(self, board=None, turn=None):
        self.action_space = np.zeros(66)
        self.player_to_move = OthelloEnv.BLACK if turn is None else turn
        self.state_size = 64
        self.prev_state = None
        if board is None:
            self.reset()
        else:
            self.state = board
            self.actions = self.possible_actions()
            result, self.done = self.check_game_over()
            self.reward = result if self.done else 0

    def step(self, action):
        self.prev_state = np.copy(self.state)

        if action == 64 or action == 65:
            observation = self.state
            result, self.done = self.check_game_over()
            self.reward = result if self.done else 0
            self.player_to_move *= -1
            self.actions = self.possible_actions()
            return observation, self.reward, self.done, self.player_to_move

        color = self.player_to_move
        self.state[action] = color

        for move in OthelloEnv.MOVE_DIRS:
            switching = []
            temp = action + OthelloEnv.MOVE_DIRS[move]

            while not self.out_of_bounds(temp, move) and self.state[temp] == -color:
                switching.append(temp)
                temp += OthelloEnv.MOVE_DIRS[move]
            
            if self.out_of_bounds(temp, move):
                pass
            
            else:
                if self.state[temp] == color:
                    for loc in switching:
                        self.state[loc] = color

        observation = self.state
        result, self.done = self.check_game_over()
        self.reward = result if self.done else 0
        self.player_to_move *= -1
        self.actions = self.possible_actions()
        return observation, self.reward, self.done, self.player_to_move        

    def pop(self):
        self.state = self.prev_state
        self.player_to_move *= -1
        self.actions = self.possible_actions()


    def out_of_bounds(self, location, move):

        def out_of_bounds_general(location):
            return location < 0 or location > 63

        def out_of_bounds_left(location):
            return location % 8 == 7
        
        def out_of_bounds_right(location):
            return location % 8 == 0


        if move == 'R' or move == 'UR' or move == 'DR':
            return out_of_bounds_right(location) or out_of_bounds_general(location)
        elif move == 'L' or move == 'UL' or move == 'DL':
            return out_of_bounds_left(location) or out_of_bounds_general(location)
        else:
            return out_of_bounds_general(location)

    def check_squares(self, location, color, board=None):        
        if board is None:
            board = self.state

        results = []
        for move in OthelloEnv.MOVE_DIRS:
            exists_move = False
            temp = location + OthelloEnv.MOVE_DIRS[move]

            while not self.out_of_bounds(temp, move) and board[temp] != color:
                if board[temp] == -color:
                    exists_move = True
                elif board[temp] == OthelloEnv.EMPTY:
                    break
                temp += OthelloEnv.MOVE_DIRS[move]

            if self.out_of_bounds(temp, move):
                exists_move = False
            if exists_move:
                if board[temp] == OthelloEnv.EMPTY:
                    results.append(temp)
            
        return results

    def possible_actions(self):
        moves = []
        for location in range(0, 64):
            color = self.state[location]
            if color == self.player_to_move:
                moves += self.check_squares(location, color)
        
        if len(moves) == 0:
            return [64] if self.player_to_move == OthelloEnv.BLACK else [65]

        return list(set(moves))

    def reset(self):
        self.state = np.zeros(64)
        self.state[27] = OthelloEnv.WHITE
        self.state[28] = OthelloEnv.BLACK
        self.state[35] = OthelloEnv.BLACK
        self.state[36] = OthelloEnv.WHITE

        self.player_to_move = OthelloEnv.BLACK
        self.actions = self.possible_actions()

        self.done = False
        self.reward = 0

    def actions_without_env(self, board, turn):
        moves = []
        for location in range(0, 64):
            color = board[location]
            if color == turn:
                moves += self.check_squares(location, color, board=board)
        
        if len(moves) == 0:
            return 64 if turn == OthelloEnv.BLACK else 65

        return list(set(moves))

    def check_game_over(self):
        if (self.state == 0).sum() == 0:
            result = self.winner()
            return result, True

        if self.actions_without_env(self.state, OthelloEnv.BLACK) == 64 and self.actions_without_env(self.state, OthelloEnv.WHITE) == 65:
            result = self.winner()
            return result, True
        
        return None, False

    def winner(self):
        return OthelloEnv.BLACK if sum(self.state) < 0 else OthelloEnv.WHITE

    def render(self):
        white = '⚪'
        black = '⚫'
        empty = '🟩'

        string = ''
        for index, token in enumerate(self.state):
            if index % 8 == 0 and index != 0:
                string += '\n'
            if token == OthelloEnv.BLACK:
                string += black + ' '
            elif token == OthelloEnv.WHITE:
                string += white + ' '
            else:
                string += empty + ' '
    
        print(string)
        return string

if __name__ == '__main__':
    env = OthelloEnv()
    env.render()

    env.step(26)
    env.render()

    env.step(18)
    env.render()
    # import random
    # choice = random.choice(env.actions)

    # observation, reward, done, turn = env.step(choice)
    # env.render()

    # while not done:
    #     print("TAKING ACTION")
    #     actions = env.actions
    #     if actions != 64 and actions != 65:
    #         action = random.choice(env.actions)
    #     else:
    #         action = actions
    #     print(action)
    #     observation, reward, done, turn = env.step(action)
    #     env.render()

    # print(done, reward)

    # env.pop()
    # env.render()
    # print('\n')
    # env.step(random.choice(env.actions))
    # env.render()