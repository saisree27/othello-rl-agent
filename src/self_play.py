from env import OthelloEnv
from agent import Agent

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
    env = OthelloEnv()
    state = env.state

    while True:
        action = agent.act(state, env.player_to_move, num)
        print(action)
        agent.add_to_memory(state, num)
        state, reward, done, turn  = env.step(action[0])

        env.render()

        if done:
            agent.update_memory(num, reward)
            return

policy_iteration(10, cont_training=True, model_file='saves/trained_model.h5')