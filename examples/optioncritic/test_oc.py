import pfrl
import torch
import torch.nn as nn
from pfrl.agents.option_critic import OptionCriticNetwork, OC
import gym
import argparse
import matplotlib.pyplot as plt



parser = argparse.ArgumentParser(description="Option Critic Example Use")
parser.add_argument('--env', default='CartPole-v0', help=('Gym Environment to run'))
parser.add_argument('--learning-rate', type=float, default=0.05, help=('Learning Rate'))
parser.add_argument('--gamma', type=float, default=.99, help=('Discount factor'))
parser.add_argument('--epsilon-start',  type=float, default=1.0, help=('Starting value for epsilon.'))
parser.add_argument('--epsilon-min', type=float, default=.1, help='Minimum epsilon.')
parser.add_argument('--epsilon-decay', type=float, default=20000, help=('Number of steps to minimum epsilon.'))
parser.add_argument('--max-history', type=int, default=10000, help=('Maximum number of steps stored in replay'))
parser.add_argument('--batch-size', type=int, default=32, help='Batch size.')
parser.add_argument('--freeze-interval', type=int, default=200, help=('Interval between target freezes.'))
parser.add_argument('--update-frequency', type=int, default=4, help=('Number of actions before each SGD update.'))
parser.add_argument('--termination-reg', type=float, default=0.01, help=('Regularization to decrease termination prob.'))
parser.add_argument('--entropy-reg', type=float, default=0.01, help=('Regularization to increase policy entropy.'))
parser.add_argument('--num-options', type=int, default=2, help=('Number of options to create.'))
parser.add_argument('--num-eps', type=int, default=1000, help=('Number of episodes to train on'))
parser.add_argument('--render', type=bool, default=False, help=('Render training episodes'))

def main(args):
    env = gym.make(args.env)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    feature_output_size = 64
    state_size = env.observation_space.shape[0]
    num_options = args.num_options
    num_actions = env.action_space.n

    featureNetwork = nn.Sequential(
        nn.Linear(state_size, 32),
        nn.ReLU(),
        nn.Linear(32, 64),
        nn.ReLU()
    )
    QNetwork = nn.Linear(feature_output_size, num_options)
    terminationNetwork = nn.Linear(feature_output_size, num_options)
    network = OptionCriticNetwork(
        featureNetwork,
        terminationNetwork,
        QNetwork,
        feature_output_size,
        num_options,
        num_actions,
        device=device,
        eps_start=args.epsilon_start,
        eps_min=args.epsilon_min,
        eps_decay=args.epsilon_decay
    )

    optim = torch.optim.RMSprop(network.parameters(), lr=args.learning_rate)
    agent = OC(
        network,
        optim,
        num_options,
        memory_size=args.max_history,
        gamma=args.gamma,
        batch_size=args.batch_size,
        freeze_interval=args.freeze_interval,
        entropy_reg=args.entropy_reg,
        termination_reg=args.entropy_reg,
        device=device
    )

    rewards = []

    for ep in range(args.num_eps):
        observation = env.reset()
        total_reward = 0
        for i in range(1000):
            if args.render:
                env.render()
            action = agent.act(observation)
            observation, reward, done, info = env.step(action)
            total_reward += reward
            agent.observe(observation, reward, done, False)
            if done:
                break
        rewards.append(total_reward)
    env.close()
    agent.save('./trained')

    plt.plot(list(range(1, args.num_eps+1)), rewards)
    plt.xlabel('Training Episode')
    plt.ylabel('Total Reward')
    plt.title('Reward over time for Option Critic')
    plt.savefig('training_graph.png', bbox_inches='tight')
    plt.show()

if __name__=="__main__":
    args = parser.parse_args()
    main(args)
