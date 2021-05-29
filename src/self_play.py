from env import OthelloEnv
from agent import Agent
from copy import deepcopy
import pickle

def policy_iteration(num_episodes, cont_training=False, model_file=None):
    print('-------STARTING POLICY ITERATION-------')
    agent = None
    if cont_training:
        agent = Agent(1, 25, model_file=model_file)
    else:
        agent = Agent(1, 25)

    while True:
        for i in range(num_episodes):
            print(f'-------EPISODE {i}-------')
            episode(agent, i)
            print('--------FINISHED---------')
            print(f'LENGTH OF AGENT MEMORY: {len(agent.memory)}')

            if len(agent.memory) > 1000:
                print('---------TRAINING--------')
                agent.replay_and_train()
                agent.save('selfplay.h5')
                print('-------MODEL SAVED-------')
                print('------SAVING MEMORY------')
                pickle.dump(agent.memory, open('selfplaymemory.obj', 'rb'))
                print('--------FINISHED---------')

def episode(agent, num):
    env = OthelloEnv()
    state = env.state
    turn = env.player_to_move
    while True:
        action = agent.act(state, env.player_to_move, num)
        # print(action)
        agent.add_to_memory(deepcopy(state), num, turn)
        state, reward, done, turn  = env.step(action[0])

        if done:
            agent.update_memory(num, reward)
            env.render()
            print("BLACK WINS!" if reward == -1 else "WHITE WINS!")
            return

policy_iteration(10, cont_training=True, model_file='saves/trained_model_corrected.h5')