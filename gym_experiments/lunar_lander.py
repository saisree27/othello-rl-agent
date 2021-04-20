# experimentation with deep Q learning with openai gym lunar lander
# partly following tutorial from https://towardsdatascience.com/reinforcement-learning-w-keras-openai-dqns-1eed3a5338c

import gym
from collections import deque, namedtuple
import tensorflow as tf
import numpy as np
import random

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        print(e)

class DQN_Agent():
    def __init__(self, env):
        self.environment = env
        self.memory = deque(maxlen=100000)
        self.gamma = 0.95          # reward depreciation
        self.epsilon = 1           # exploration / exploitation
        self.epsilon_min = 0.01    # always explore 1 percent of the time
        self.epsilon_decay = 0.995 # at each time step, decrease epsilon
        self.lr = 0.01             # network learning rate
        self.input_shape = self.environment.observation_space.shape
        self.action_space = self.environment.action_space
        self.model = self.neural_net()
        self.batch_size = 64

    def neural_net(self):
        model = tf.keras.models.Sequential()
        
        model.add(tf.keras.layers.Dense(100, activation="relu", input_shape=(self.input_shape[0],)))
        model.add(tf.keras.layers.Dense(120, activation="relu"))
        model.add(tf.keras.layers.Dense(100, activation="relu"))
        model.add(tf.keras.layers.Dense(self.action_space.n))

        model.compile(optimizer=tf.keras.optimizers.Adam(lr=self.lr), loss='mse')

        return model

    def add_to_memory(self, state, action, reward, new_state, done):
        self.memory.append( (state, action, reward, new_state, done) )
    
    def replay_and_train(self):
        if len(self.memory) < self.batch_size:
            return

        sample = random.sample(self.memory, self.batch_size)
        
        states = np.array([i[0] for i in sample])
        actions = np.array([i[1] for i in sample])
        rewards = np.array([i[2] for i in sample])
        next_states = np.array([i[3] for i in sample])
        dones = np.array([i[4] for i in sample])

        states = np.squeeze(states)
        next_states = np.squeeze(next_states)

        next_states_predicted_q = np.argmax(self.model.predict_on_batch(next_states)[0])

        to_train_on = rewards + self.gamma * next_states_predicted_q * (1 - dones)
        predictions = self.model.predict_on_batch(states)

        indices = np.array([i for i in range(self.batch_size)])
        predictions[[indices], [actions]] = to_train_on

        self.model.fit(states, predictions, epochs=1, verbose=0)

    def act(self, state):
        self.epsilon *= self.epsilon_decay
        self.epsilon = max(self.epsilon_min, self.epsilon)

        if np.random.random() < self.epsilon:
            return self.action_space.sample()
        else:
            return np.argmax(self.model.predict(state)[0])

env = gym.make('LunarLander-v2')
env.reset()

NUM_EPISODES = 100
NUM_TIME_STEPS = 1000

agent = DQN_Agent(env)

print("STARTING TRAINING")
print("------------------------")

for episode in range(NUM_EPISODES):
    state = env.reset().reshape(1, 8)
    print(f"EPISODE {episode}")
    score = 0

    for t in range(NUM_TIME_STEPS):
        action = agent.act(state)
        env.render()

        new_state, reward, done, info = env.step(action)
        score += reward
        
        new_state = new_state.reshape(1, 8)

        agent.add_to_memory( state, action, reward, new_state, done )
        agent.replay_and_train()

        state = new_state

        if done:
            print(f"Episode finished after {t} timesteps, score: {score}")
            break

env.close()