import pfrl
import torch
import torch.nn as nn
from pfrl.agents.option_critic import OptionCriticNetwork, OC
import gym

def main():
        env = gym.make('CartPole-v0')
        observation = env.reset()

        feature_output_size = 32
        state_size = 4
        num_options = 8
        num_actions = 2
        featureNetwork = nn.Linear(state_size, feature_output_size)
        QNetwork = nn.Linear(feature_output_size, num_options)
        terminationNetwork = nn.Linear(feature_output_size, num_options)
        network = OptionCriticNetwork(featureNetwork, terminationNetwork, QNetwork, feature_output_size, num_options, num_actions)

        optim = torch.optim.RMSprop(network.parameters())
        agent = OC(network, optim, 8)

        for i in range(1000):
            env.render()
            action = agent.act(observation)
            observation, reward, done, info = env.step(action)
            agent.observe(observation, reward, done, False)
            #if done:
            #    break
        env.close()

if __name__=="__main__":
        main()
