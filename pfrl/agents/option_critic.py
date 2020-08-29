# Adapted from https://github.com/lweitkamp/option-critic-pytorch

import torch
import torch.nn as nn
from torch.distributions import Categorical, Bernoulli
from pfrl import agent

class OptionCriticNetwork(nn.Module):
    """Option Critic Network

    This class is used to instantiate the Option Critic Architecture.

    Args:
        featureNetwork (torch.nn.Module): Network to convert from state to features
        terminationNetwork (torch.nn.Module): Network to determine termination probs
        QNetwork (torch.nn.Module): Network to calculate Q-Value
        feature_output_size (int): size of feature vector outputted by featureNetwork
        device (string): device to put network on
        eps_start (float): starting epsilon
        eps_min (float): minimum epsilon
        eps_decay (int): decay rate for epsilon

    """
    def __init__(self, featureNetwork, terminationNetwork, QNetwork, feature_output_size, device='cpu',
                 eps_start = 1.0, eps_min = 0.1, eps_decay = int(1e6)):
        super(OptionCriticNetwork, self).__init__()

        self.eps_start = eps_start
        self.eps_min = eps_min
        self.eps_decay = eps_decay
        self.num_steps = 0

        self.features = featureNetwork
        self.terminations = terminationNetwork
        self.Q = QNetwork
        self.options_W = nn.Parameter(torch.zeros(num_options, feature_output_size, num_actions))
        self.options_b = nn.Parameter(torch.zeros(num_options, num_actions))

        self.to(device)

    def get_state(self, obs):
        """Convert state to features."""
        obs = obs.to(self.device)
        state = self.features(obs)
        return state

    def get_Q(self, state):
        """Get Q value from state."""
        return self.Q(state)

    def predict_option_termination(self, state, current_option):
        """Predict whether option will terminate, return (termination, next option)."""
        termination = self.terminations(state)[:, current_option].sigmoid()
        option_termination = Bernoulli(termination).sample()

        Q = self.get_Q(state)
        next_option = Q.argmax(dim=-1)
        return bool(option_termination.item()), next_option.item()

    def get_terminations(self, state):
        """Get termination probabilities."""
        return self.terminations(state).sigmoid()

    def get_action(self, state, option):
        """Get action based on state and current option."""
        logits = state @ self.options_W[option] + self.options_b[option]
        action_dist = logits.softmax(dim=-1)
        action_dist = Categorical(action_dist)

        action = action_dist.sample()
        logp = action_dist.log_prob(action)
        entropy = action_dist.entropy()

        return action.item(), logp, entropy

    def greedy_option(self, state):
        """Get greedy option."""
        Q = self.get_Q(state)
        return Q.argmax(dim=-1).item()

    @property
    def epsilon(self):
        """Return epsilon value"""
        eps = self.eps_min + (self.eps_start - self.eps_min) * exp(-self.num_steps / self.eps_decay)
        self.num_steps += 1

        return eps

class OC(agent.Agent):
    """Option Critic Architecture

    Args:
        oc (OptionCriticNetwork): Option Critic Network
        optimizer (Optimizer): Already set up optimizer

    """

    def __init__(self, oc, optimizer):
        self.oc = oc

    def act(self, obs):
        print(obs)

    def observe(self, obs, reward, done, reset):
        print(reward)