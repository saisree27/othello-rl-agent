from env import OthelloEnv
from agent import Agent
from mcts import MCTS
from copy import deepcopy

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
            print(f'--------FINISHED---------')
            if len(agent.memory) > 10000:
                print(f'---------TRAINING--------')
                agent.replay_and_train()
                agent.save('BestModel.h5')
                print(f'-------MODEL SAVED-------')

def episode(agent, num):
    state = agent.env.state
    agent.build_new_MCTS()

    while True:
        action = agent.act()
        agent.add_to_memory( (state, agent.mcts.pi(state), None, num) )
        state, reward, done, turn  = agent.env.step(action)
        
        if done:
            agent.update_memory(num, reward)
            return
