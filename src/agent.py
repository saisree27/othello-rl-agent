from env import OthelloEnv
from mcts  import MCTS # TODO
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model, Model
from tensorflow.keras.layers import Input, Dense, Activation, Flatten, BatchNormalization, Conv2D, LeakyReLU
from tensorflow.keras.losses import CategoricalCrossentropy
import random

class Agent():
    def __init__(self, cpuct, sims, model_file=None, env=OthelloEnv(), memory=None):
        self.model = load_model(model_file) if model_file is not None else get_model()
        self.env = env
        self.cpuct = cpuct
        self.sims = sims
        self.MCTS = None # MCTS is built in build_new_MCTS()
        self.memory = [] if memory is None else memory
        self.epochs = 50
        self.batch_size = 100

    def get_model(self):
        # random model from connect4 example, optimize later

        board_input = Input(shape=(8,8), name='board')
        x = Conv2D(filters=128, kernel_size=(4,4))(board_input)
        x = BatchNormalization(axis=1)(x)
        x = LeakyReLU()(x)
        x = Conv2D(filters=128, kernel_size=(4,4))(board_input)
        x = BatchNormalization(axis=1)(x)
        x = LeakyReLU()(x)
        x = Conv2D(filters=128, kernel_size=(4,4))(board_input)
        x = BatchNormalization(axis=1)(x)
        x = LeakyReLU()(x)

        y_policy = Conv2D(filters=2, kernel_size=(1,1))(x)
        y_policy = BatchNormalization(axis=1)(y_policy)
        y_policy = LeakyReLU()(y_policy)
        y_policy = Flatten()(y_policy)
        y_policy = Dense(64, name='policy_output')(y_policy)
        
        y_val = Conv2D(filters=1, kernel_size=(1,1))(x)
        y_val = BatchNormalization(axis=1)(y_val)
        y_val = LeakyReLU()(y_val)
        y_val = Flatten()(y_val)
        y_val = Dense(1, name='value_output')(y_val)

        model = Model(inputs=[board_input], outputs=[y_policy, y_val])
        model.compile(loss={'value_output': 'mean_squared_error', 'policy_output': CategoricalCrossentropy},
			optimizer=tf.keras.optimizers.SGD(lr=self.learning_rate),	
			loss_weights={'value_head': 0.5, 'policy_head': 0.5}	
			)

        return model
    
    def act(self, state, turn, episode_num):
        if self.mcts is None:
            self.build_new_MCTS(state)
        else:
            self.change_MCTS_root(state)

        for simulation in range(sims):
            temp_env = OthelloEnv(state, turn)
            # self.MCTS.search(temp_env, self.model) does search
            # self.MCTS.pi is resulting policy
        self.memory.append( (state, self.MCTS.pi(state), None, episode_num) ) # episode_num used to update rewards later
        action = random.choices(len(self.MCTS.pi(state)), weights=self.MCTS.pi(s), k=1)
        return action
    
    def update_memory(episode_num, reward):
        for i in range(len(self.memory)):
            if self.memory[i][3] == episode_num:
                self.memory[i][2] = reward
        return

    def replay_and_train(self):
        for epoch in range(self.epochs):
            batch = random.sample(self.memory, self.batch_size)
            X = np.array([b[0] for b in batch])
            Y_pi = np.array(b[1] for b in batch)
            Y_val = np.array(b[2] for b in batch)

            Y = {'policy_output': Y_pi, 'value_output': Y_val}

            self.model.fit(X, Y, epochs=1, verbose=True)
        return
    
    def build_new_MCTS(self, state):
        '''
        Builds new MCTS tree and assigns to self.MCTS
        '''
        self.MCTS = MCTS(state)
        return
    
    def change_MCTS_root(self, state):
        '''
        Changes root node of self.MCTS
        '''
        return