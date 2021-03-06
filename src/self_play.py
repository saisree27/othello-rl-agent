from env import OthelloEnv
from agent import Agent
from copy import deepcopy
import pickle
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

def policy_iteration(num_episodes, cont_training=False, model_file=None):
    print('-------STARTING POLICY ITERATION-------')
    agent = None
    memory = pickle.load(open(MEMORY_BEGIN, 'rb'))
    # memory = []
    if cont_training:
        agent = Agent(0.5, 25, model_file=model_file, memory=memory)
    else:
        agent = Agent(0.5, 25, memory=memory)

    print(agent.model.summary())

    print(f'LENGTH OF AGENT MEMORY: {len(agent.memory)}')

    for i in range(num_episodes):
        print(f'---------------------EPISODE {i}-------------------------')
        episode(agent, i)
        print('----------------------FINISHED-----------------------------')
        print(f'LENGTH OF AGENT MEMORY: {len(agent.memory)}')

        if len(agent.memory) > 3000:
            print('----------------------TRAINING-------------------------------------')
            agent.replay_and_train()
            agent.save(SAVE_FILE)
            print('--------------------MODEL SAVED--------------------------------------')
            print('-------------------------SAVING MEMORY------------------------')
            pickle.dump(agent.memory, open(MEMORY_SAVE, 'wb'))
            print('-----------------------------FINISHED-------------------------------')

def episode(agent, num):
    env = OthelloEnv()
    state = env.state
    turn = env.player_to_move
    while True:
        action = agent.act(state, env.player_to_move, num)
        # print(action)
        agent.add_to_memory(deepcopy(state), num, turn)
        state, reward, done, turn  = env.step(action[0])
        
        env.render()
        print('\n')

        if done:
            agent.update_memory(num, reward)
            env.render()
            print("BLACK WINS!" if reward == -1 else "WHITE WINS!")
            return


BEGIN_TRAINING = 'saves/large_model_trained_database.h5'
SAVE_FILE = 'saves/large_model_self_play.h5'
MEMORY_BEGIN = 'saves/8k_decent_memory.obj'
MEMORY_SAVE = 'saves/large_model_memory_post_8k.obj'

policy_iteration(1000000, cont_training=True, model_file=BEGIN_TRAINING)