from env import OthelloEnv
from agent import Agent
import random
import tensorflow as tf
import numpy as np
from agents.alpha_beta import AlphaBeta
from agents.minimax import MinimaxOthelloRunner
from copy import deepcopy

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)

def play_against_minimax(agent, color):
    # TODO
    return

def play_against_alpha_beta(agent, color):
    game = OthelloEnv()
    done = False
    turn = OthelloEnv.BLACK
    reward = 0
    while not done:
        if turn == color:
            move = agent.act(game.state, game.player_to_move, -99)
            move = move[0]
            print(move)
            _, reward, done, turn = game.step(move)
        else:
            alphabeta = AlphaBeta(OthelloEnv(deepcopy(game.state), turn))
            move = alphabeta.play()

            _, reward, done, turn = game.step(move)
        game.render()
        print('\n')
    return reward

def play_against_random(agent, color):
    game = OthelloEnv()
    done = False
    turn = OthelloEnv.BLACK
    reward = 0
    while not done:
        if turn == color:
            move = agent.act(game.state, game.player_to_move, -99)
            move = move[0]
            print(move)
            _, reward, done, turn = game.step(move)
        else:
            print("RANDOM MOVE")
            _, reward, done, turn = game.step(random.choice(game.actions))
        game.render()
        print('\n')
    
    return reward


agent = Agent(model_file='saves/18k_post_0.4valueloss.h5', cpuct=0, sims=50, deterministic=True)
print(agent.model.summary())

print('#1: AlphaBeta')
print('#2: Random')
print('#3: Minimax (not yet implemented)')

opponent = int(input('What should the agent play against? '))
color = int(input('Should the agent play White (1) or Black (-1)? '))

if opponent == 1:
    print('Running game against alphabeta.')
    res = play_against_alpha_beta(agent, color)
    print(res)


if opponent == 2:
    print('Running game against random.')
    res = play_against_random(agent, color)
    print(res)

if opponent == 3:
    print('Not yet implemented.')

