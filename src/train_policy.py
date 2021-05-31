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

BATCH_SIZE = 2048
EPOCHS = 10

training_data = np.load('saves/training_data_corrected.npy', allow_pickle=True)

agent = Agent(large_model=True)
print(agent.model.summary())

for epoch in range(EPOCHS):
    print(f'EPOCH {epoch}')
    batch = np.random.choice(training_data.shape[0], BATCH_SIZE)
    batch = training_data[batch]

    X = np.array([b[0] * b[3] for b in batch])

    new_X = []

    for board in X:
        new_board = np.reshape(board,  (1, 8, 8))
        new_X.append(new_board)
    
    X = np.array(new_X)
    Y_pi = np.array([b[1] for b in batch])
    Y_val = np.array([b[2] * b[3] for b in batch])


    X = np.asarray(X).astype('float32')
    Y_pi = np.asarray(Y_pi).astype('float32')
    Y_val = np.asarray(Y_val).astype('float32')

    print(Y_val)
    print(Y_pi)

    Y = {'policy_output': Y_pi, 'value_output': Y_val}
    agent.model.fit(X, Y, epochs=1, verbose=True, batch_size=1)


agent.save('saves/large_model_trained_database.h5')