from env import OthelloEnv
from agent import Agent
import random
import tensorflow as tf
import numpy as np
from agents.alpha_beta import AlphaBeta
from agents.minimax import MinimaxOthelloRunner

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
            # change this to agent.act once mcts finished
            # literally just using human-like policy and no simulations, so this does not do very well
            model_input = agent.convert_to_model_input([game.state])
            policy, _ = agent.model.predict(model_input)
            mask = np.zeros(66)
            for x in game.actions:
                mask[x] = 1

            policy = policy[0]

            policy = policy * mask
        
            move = np.argmax(policy)
            print(move)
            _, reward, done, turn = game.step(move)
        else:
            alphabeta = AlphaBeta(OthelloEnv(game.state, turn))
            move = alphabeta.play()

            _, reward, done, turn = game.step(move)
        game.render()

def play_against_random(agent, color):
    game = OthelloEnv()
    done = False
    turn = OthelloEnv.BLACK
    reward = 0
    while not done:
        if turn == color:
            # change this to agent.act once mcts finished
            # literally just using human-like policy and no simulations, so this does not do very well
            model_input = agent.convert_to_model_input([game.state])
            policy, _ = agent.model.predict(model_input)
            mask = np.zeros(66)
            for x in game.actions:
                mask[x] = 1

            policy = policy[0]

            policy = policy * mask
        
            move = np.argmax(policy)
            print(move)
            _, reward, done, turn = game.step(move)
        else:
            _, reward, done, turn = game.step(random.choice(game.actions))
        game.render()
    
    return reward


agent = Agent(model_file='saves/trained_model.h5')

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

