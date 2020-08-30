# Adapted from https://github.com/lweitkamp/option-critic-pytorch

import torch
import torch.nn as nn
import numpy as np
import random
from math import exp
from torch.distributions import Categorical, Bernoulli
from pfrl import agent
from pfrl.replay_buffers.hrl_replay_buffer import OptionCriticReplayBuffer
from copy import deepcopy

class OptionCriticNetwork(nn.Module):
    """Option Critic Network

    This class is used to instantiate the Option Critic Architecture.

    Args:
        featureNetwork (torch.nn.Module): Network to convert from state to features
        terminationNetwork (torch.nn.Module): Network to determine termination probs
        QNetwork (torch.nn.Module): Network to calculate Q-Value
        feature_output_size (int): size of feature vector outputted by featureNetwork
        num_options (int): Number of options used
        num_actions (int): Number of actions available
        device (string): device to put network on
        eps_start (float): starting epsilon
        eps_min (float): minimum epsilon
        eps_decay (int): decay rate for epsilon

    """
    def __init__(self, featureNetwork, terminationNetwork, QNetwork, feature_output_size, num_options, num_actions, device='cpu',
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

        self.device = device

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
        termination = self.terminations(state)[current_option].sigmoid()
        option_termination = Bernoulli(termination).sample()

        return bool(option_termination.item())

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

    def __init__(
        self,
        oc,
        optimizer,
        num_options,
        memory_size=10000,
        gamma=0.99,
        entropy_reg=0.01,
        termination_reg=0.01
    ):
        self.oc = oc
        self.buffer = OptionCriticReplayBuffer(capacity=memory_size)
        self.gamma = gamma
        self.entropy_reg = entropy_reg
        self.termination_reg = termination_reg
        self.oc_prime = deepcopy(oc)
        self.optimizer = optimizer
        self.option = 0
        self.option_termination = True
        self.num_options = num_options


    def act(self, obs):
        obs = self.to_tensor(obs)
        self.prev_obs = obs
        state = self.oc.get_state(obs)
        if self.option_termination:
            epsilon = self.oc.epsilon
            greedy_option = self.oc.greedy_option(state)
            self.option = np.random.choice(self.num_options) if np.random.rand() < epsilon else greedy_option
        action, logp, entropy = self.oc.get_action(state, self.option)
        self.logp = logp
        self.entropy = entropy

        return action

    def observe(self, obs, reward, done, reset):
        obs = self.to_tensor(obs)
        state = self.oc.get_state(obs)
        self.option_termination = self.oc.predict_option_termination(state, self.option)

        actor_loss = self.actor_loss_fn(self.prev_obs, self.option, self.logp, self.entropy, reward, done, obs)
        return

    def load(self):
        return

    def save(self):
        return

    def get_statistics(self):
        return

    def to_tensor(self, arr):
        arr = np.array(arr)
        arr = torch.from_numpy(arr).float()
        return arr


    def actor_loss_fn(self, obs, option, logp, entropy, reward, done, next_obs):
        state = self.oc.get_state(self.to_tensor(obs))
        next_state = self.oc.get_state(self.to_tensor(next_obs))
        next_state_prime = self.oc_prime.get_state(self.to_tensor(next_obs))

        option_term_prob = self.oc.get_terminations(state)[option]
        next_option_term_prob = self.oc.get_terminations(next_state)[option]

        Q = self.oc.get_Q(state).detach().squeeze()
        next_Q_prime = self.oc_prime.get_Q(next_state_prime).detach().squeeze()

        gt = reward + (1-done) * self.gamma * \
            ((1-next_option_term_prob) * next_Q_prime[option] + next_option_term_prob * next_Q_prime.max(dim=-1)[0])

        termination_loss = option_term_prob * \
            (Q[option].detach() - Q.max(dim=-1)[0].detach() + self.termination_reg) * (1-done)

        policy_loss = -logp * (gt.detach() - Q[option]) - self.entropy_reg * entropy

        actor_loss = termination_loss + policy_loss

        return actor_loss
