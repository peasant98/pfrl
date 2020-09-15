import pfrl
import torch
import torch.nn as nn
from pfrl.agents.option_critic import OptionCriticNetwork, OC
import gym
import argparse

def main():




    env = gym.make('CartPole-v0')
    num_eps = 1000

    feature_output_size = 32
    state_size = 4
    num_options = 2
    num_actions = 2
    featureNetwork = nn.Linear(state_size, feature_output_size)
    QNetwork = nn.Linear(feature_output_size, num_options)
    terminationNetwork = nn.Linear(feature_output_size, num_options)
    network = OptionCriticNetwork(featureNetwork, terminationNetwork, QNetwork, feature_output_size, num_options, num_actions)

    optim = torch.optim.RMSprop(network.parameters(), lr=0.005)
    agent = OC(network, optim, num_options)

    for ep in range(num_eps):
        observation = env.reset()
        total_reward = 0
        for i in range(1000):
            env.render()
            action = agent.act(observation)
            observation, reward, done, info = env.step(action)
            total_reward += reward
            agent.observe(observation, reward, done, False)
            if done:
                break
        print(total_reward)
    env.close()

if __name__=="__main__":
    main()
