from env import OthelloEnv
import numpy as np
from tqdm import tqdm
from copy import deepcopy

squares = {
    'A1': 0,  'A2': 1,  'A3': 2,  'A4': 3,  'A5': 4,  'A6': 5,  'A7': 6,  'A8': 7,
    'B1': 8,  'B2': 9,  'B3': 10, 'B4': 11, 'B5': 12, 'B6': 13, 'B7': 14, 'B8': 15,
    'C1': 16, 'C2': 17, 'C3': 18, 'C4': 19, 'C5': 20, 'C6': 21, 'C7': 22, 'C8': 23,
    'D1': 24, 'D2': 25, 'D3': 26, 'D4': 27, 'D5': 28, 'D6': 29, 'D7': 30, 'D8': 31,
    'E1': 32, 'E2': 33, 'E3': 34, 'E4': 35, 'E5': 36, 'E6': 37, 'E7': 38, 'E8': 39,
    'F1': 40, 'F2': 41, 'F3': 42, 'F4': 43, 'F5': 44, 'F6': 45, 'F7': 46, 'F8': 47,
    'G1': 48, 'G2': 49, 'G3': 50, 'G4': 51, 'G5': 52, 'G6': 53, 'G7': 54, 'G8': 55,
    'H1': 56, 'H2': 57, 'H3': 58, 'H4': 59, 'H5': 60, 'H6': 61, 'H7': 62, 'H8': 63
}

def moves(s):
    for x in range(0, len(s), 2):
        yield squares[s[x:x+2]]

def process_txt(filename):
    dataset = []
    lines = open(filename,'r').readlines()
    lines = [line.strip() for line in lines]
    for i, line in tqdm(enumerate(lines)):
        if line is not None:
            moves_list = moves(line)
            env, done, reward = OthelloEnv(), False, 0
            state = env.state
            for move in moves_list:
                if env.actions[0] in [64, 65]:
                    policy = np.zeros(66)
                    policy[env.actions[0]] = 1
                    dataset.append( [state, policy, None, i, env.player_to_move] )
                    state, reward, done, _ = env.step(env.actions[0])

                if not done:
                    policy = np.zeros(66)
                    policy[move] = 1
                    dataset.append( [state, policy, None, i, env.player_to_move] )
                    state, reward, done, _ =  env.step(move)
                
                if done:
                    for entry in dataset:
                        if entry[3] == i:
                            entry[2] = reward
    
    np_dataset = np.array([np.array([state, policy, reward, turn]) for state, policy, reward, _, turn in dataset])
    return np_dataset

dataset = process_txt('saves\WTH_2018.txt')
print(dataset.shape)

print(dataset[0])

np.save('saves/training_data_corrected.npy', dataset)