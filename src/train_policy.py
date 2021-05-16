from tensorflow.python.keras.backend import one_hot
from env import OthelloEnv
from agent import Agent
import numpy as np
import tensorflow as tf


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

BATCH_SIZE = 500
EPOCHS = 25

training_data = np.load('training_data.npy', allow_pickle=True)

agent = Agent()

for epoch in range(EPOCHS):
    print(f'EPOCH {epoch}')
    batch = np.random.choice(training_data.shape[0], BATCH_SIZE)
    batch = training_data[batch]

    X = np.array([b[0] for b in batch])

    new_X = []

    for board in X:
        one_hot_encoded_black = np.zeros(64)
        one_hot_encoded_white = np.zeros(64)
        # print(board)
        for i, piece in enumerate(board):
            if piece < 0:
                one_hot_encoded_black[i] = 1
            elif piece > 0:
                one_hot_encoded_white[i] = 1
        
        one_hot_encoded_black = np.reshape( one_hot_encoded_black, (8,8) )
        one_hot_encoded_white = np.reshape( one_hot_encoded_white, (8,8) )

        new_board = np.reshape(np.append(one_hot_encoded_black, one_hot_encoded_white), (2, 8, 8))

        new_X.append(new_board)
    
    X = np.array(new_X)
    Y_pi = np.array([b[1] for b in batch])
    Y_val = np.array([b[2] for b in batch])


    X = np.asarray(X).astype('float32')
    Y_pi = np.asarray(Y_pi).astype('float32')
    Y_val = np.asarray(Y_val).astype('float32')

    Y = {'policy_output': Y_pi, 'value_output': Y_val}
    agent.model.fit(X, Y, epochs=1, verbose=True, batch_size=1)


agent.save('trained_model.h5')