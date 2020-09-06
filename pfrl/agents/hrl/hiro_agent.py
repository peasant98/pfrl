import torch
from torch import nn
import numpy as np
import os

import pfrl
from pfrl.agent import HRLAgent
from pfrl.replay_buffers import (
    LowerControllerReplayBuffer,
    HigherControllerReplayBuffer
)
from pfrl import explorers
from pfrl.replay_buffer import high_level_batch_experiences_with_goal
from pfrl.agents import HIROGoalConditionedTD3


def _is_update(episode, freq, ignore=0, rem=0):
    if episode != ignore and episode % freq == rem:
        return True
    return False


def _mean_or_nan(xs):
    """Return its mean a non-empty sequence, numpy.nan for a empty one."""
    return np.mean(xs) if xs else np.nan

# standard controller


class ConstantsMult(nn.Module):
    def __init__(self, constants):
        super().__init__()
        self.constants = constants

    def forward(self, x):
        return self.constants * x


class HRLControllerBase():
    def __init__(
            self,
            state_dim,
            goal_dim,
            action_dim,
            scale,
            model_path,
            replay_buffer,
            name='controller_base',
            actor_lr=0.0001,
            critic_lr=0.001,
            expl_noise=0.1,
            policy_noise=0.2,
            noise_clip=0.5,
            gamma=0.99,
            policy_freq=2,
            tau=0.005,
            replay_start_size=100,
            is_low_level=True,
            buffer_freq=10,
            minibatch_size=100,
            gpu=None):
        # example name- 'td3_low' or 'td3_high'
        self.name = name
        self.scale = scale
        self.model_path = model_path
        # parameters
        self.expl_noise = expl_noise
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.gamma = gamma
        self.policy_freq = policy_freq
        self.tau = tau
        self.is_low_level = is_low_level
        self.minibatch_size = minibatch_size
        # create td3 agent
        self.device = torch.device(f"cuda:{gpu}")

        policy = nn.Sequential(
            nn.Linear(state_dim + goal_dim, 300),
            nn.ReLU(),
            nn.Linear(300, 300),
            nn.ReLU(),
            nn.Linear(300, action_dim),
            nn.Tanh(),
            ConstantsMult(torch.tensor(self.scale).float().to(self.device)),
            pfrl.policies.DeterministicHead(),
            )
        policy_optimizer = torch.optim.Adam(policy.parameters(), lr=actor_lr)

        def make_q_func_with_optimizer():
            q_func = nn.Sequential(
                pfrl.nn.ConcatObsAndAction(),
                nn.Linear(state_dim + goal_dim + action_dim, 300),
                nn.ReLU(),
                nn.Linear(300, 300),
                nn.ReLU(),
                nn.Linear(300, 1),
            )
            q_func_optimizer = torch.optim.Adam(q_func.parameters(), lr=critic_lr)
            return q_func, q_func_optimizer

        q_func1, q_func1_optimizer = make_q_func_with_optimizer()
        q_func2, q_func2_optimizer = make_q_func_with_optimizer()

        # TODO - have proper low and high values from action space.
        explorer = explorers.AdditiveGaussian(
            scale=1.0
        )

        self.agent = HIROGoalConditionedTD3(
            policy,
            q_func1,
            q_func2,
            policy_optimizer,
            q_func1_optimizer,
            q_func2_optimizer,
            replay_buffer,
            gamma=gamma,
            soft_update_tau=tau,
            explorer=explorer,
            update_interval=policy_freq,
            replay_start_size=replay_start_size,
            is_low_level=self.is_low_level,
            buffer_freq=buffer_freq,
            minibatch_size=minibatch_size,
            gpu=gpu
        )
        self.device = self.agent.device

        self._initialized = False
        self.total_it = 0

    def save(self, directory):
        """
        save the internal state of the TD3 agent.
        """
        self.agent.save(directory)

    def load(self, directory):
        """
        load the internal state of the TD3 agent.
        """
        self.agent.load(directory)

    def policy(self, state, goal):
        """
        run the policy (actor).
        """
        return self.agent.act_with_goal(torch.FloatTensor(state), torch.FloatTensor(goal))

    def _observe(self, states, goals, rewards, done, state_arr=None, action_arr=None):
        """
        observe, and train (if we can sample from the replay buffer)
        """
        self.agent.observe_with_goal(torch.FloatTensor(states), torch.FloatTensor(goals), rewards, done, None)

    def observe(self, states, goals, rewards, done, iterations=1):
        """
        get data from the replay buffer, and train.
        """
        return self._observe(states, goals, rewards, goals, done)


# lower controller
class LowerController(HRLControllerBase):
    def __init__(
            self,
            state_dim,
            goal_dim,
            action_dim,
            scale,
            model_path,
            replay_buffer,
            name='lower_controller',
            actor_lr=0.0001,
            critic_lr=0.001,
            expl_noise=1.0,
            policy_noise=0.2,
            noise_clip=0.5,
            gamma=0.99,
            policy_freq=2,
            tau=0.005,
            is_low_level=True,
            minibatch_size=100,
            gpu=None):
        super(LowerController, self).__init__(
                                            state_dim=state_dim,
                                            goal_dim=goal_dim,
                                            action_dim=action_dim,
                                            scale=scale,
                                            model_path=model_path,
                                            replay_buffer=replay_buffer,
                                            name=name,
                                            actor_lr=actor_lr,
                                            critic_lr=critic_lr,
                                            expl_noise=expl_noise,
                                            policy_noise=policy_noise,
                                            noise_clip=noise_clip,
                                            gamma=gamma,
                                            policy_freq=policy_freq,
                                            tau=tau,
                                            is_low_level=is_low_level,
                                            minibatch_size=minibatch_size,
                                            gpu=gpu)
        self.name = name

    def observe(self, n_s, g, r, done):

        return self._observe(n_s, g, r, done)


# higher controller

class HigherController(HRLControllerBase):
    def __init__(
            self,
            state_dim,
            goal_dim,
            action_dim,
            scale,
            model_path,
            replay_buffer,
            name='higher_controller',
            actor_lr=0.0001,
            critic_lr=0.001,
            expl_noise=0.1,
            policy_noise=0.2,
            noise_clip=0.5,
            gamma=0.99,
            policy_freq=2,
            tau=0.005,
            is_low_level=False,
            buffer_freq=10,
            minibatch_size=100,
            gpu=None):
        super(HigherController, self).__init__(
                                                state_dim=state_dim,
                                                goal_dim=goal_dim,
                                                action_dim=action_dim,
                                                scale=scale,
                                                model_path=model_path,
                                                name=name,
                                                replay_buffer=replay_buffer,
                                                actor_lr=actor_lr,
                                                critic_lr=critic_lr,
                                                expl_noise=expl_noise,
                                                policy_noise=policy_noise,
                                                noise_clip=noise_clip,
                                                gamma=gamma,
                                                policy_freq=policy_freq,
                                                tau=tau,
                                                is_low_level=is_low_level,
                                                buffer_freq=buffer_freq,
                                                minibatch_size=minibatch_size,
                                                gpu=gpu)
        self.name = 'high'
        self.action_dim = action_dim

    def off_policy_corrections(self, low_con, batch_size, sgoals, states, actions, candidate_goals=8):
        """
        implementation of the novel off policy correction in the HIRO paper.
        """

        first_s = [s[0] for s in states]  # First x
        last_s = [s[-1] for s in states]  # Last x

        # Shape: (batch_size, 1, subgoal_dim)
        # diff = 1
        # different in goals
        diff_goal = (np.array(last_s) -
                     np.array(first_s))[:, np.newaxis, :self.action_dim]

        # Shape: (batch_size, 1, subgoal_dim)
        # original = 1
        # random = candidate_goals
        original_goal = np.array(sgoals)[:, np.newaxis, :]
        # select random goals
        random_goals = np.random.normal(loc=diff_goal, scale=.5*self.scale[None, None, :],
                                        size=(batch_size, candidate_goals, original_goal.shape[-1]))
        random_goals = random_goals.clip(-self.scale, self.scale)

        # Shape: (batch_size, 10, subgoal_dim)
        candidates = np.concatenate([original_goal, diff_goal, random_goals], axis=1)
        # states = np.array(states)[:, :-1, :]
        actions = np.array(actions)
        seq_len = len(states[0])

        # For ease
        new_batch_sz = seq_len * batch_size
        action_dim = actions[0][0].shape
        obs_dim = states[0][0].shape
        ncands = candidates.shape[1]

        true_actions = actions.reshape((new_batch_sz,) + action_dim)
        observations = states.reshape((new_batch_sz,) + obs_dim)
        goal_shape = (new_batch_sz, self.action_dim)
        # observations = get_obs_tensor(observations, sg_corrections=True)

        # batched_candidates = np.tile(candidates, [seq_len, 1, 1])
        # batched_candidates = batched_candidates.transpose(1, 0, 2)

        policy_actions = np.zeros((ncands, new_batch_sz) + action_dim)

        for c in range(ncands):
            subgoal = candidates[:,c]
            candidate = (subgoal + states[:, 0, :self.action_dim])[:, None] - states[:, :, :self.action_dim]
            candidate = candidate.reshape(*goal_shape)
            policy_actions[c] = low_con.policy(torch.tensor(observations).float(), torch.tensor(candidate).float())

        difference = (policy_actions - true_actions)
        difference = np.where(difference != -np.inf, difference, 0)
        difference = difference.reshape((ncands, batch_size, seq_len) + action_dim).transpose(1, 0, 2, 3)

        logprob = -0.5*np.sum(np.linalg.norm(difference, axis=-1)**2, axis=-1)
        max_indices = np.argmax(logprob, axis=-1)
        # return best candidates with maximum probability
        return candidates[np.arange(batch_size), max_indices]

    def observe(self, low_con, state_arr, action_arr, r, g, n_s, done):
        """
        train the high level controller with
        the novel off policy correction.
        """
        # step 1 - record experience in replay buffer

        self.agent.observe_with_goal_state_action_arr(torch.FloatTensor(state_arr),
                                                      torch.FloatTensor(action_arr),
                                                      torch.FloatTensor(n_s),
                                                      torch.FloatTensor(g), r, done, None)

        # step 2 - if we can update, sample from replay buffer first
        batch = self.agent.sample_if_possible()
        if batch:
            experience = high_level_batch_experiences_with_goal(batch, self.device, lambda x: x, self.gamma)
            actions = experience['action']
            action_arr = experience['action_arr']
            state_arr = experience['state_arr']
            actions = self.off_policy_corrections(
                low_con,
                self.minibatch_size,
                actions.cpu().data.numpy(),
                state_arr.cpu().data.numpy(),
                action_arr.cpu().data.numpy())

            tensor_actions = torch.FloatTensor(actions).to(self.agent.device)
            # relabel actions
            experience['action'] = tensor_actions

            self.agent.high_level_update_batch(experience)


class HIROAgent(HRLAgent):
    def __init__(self,
                 state_dim,
                 action_dim,
                 goal_dim,
                 subgoal_dim,
                 subgoal_space,
                 scale_low,
                 start_training_steps,
                 model_save_freq,
                 model_path,
                 buffer_size,
                 batch_size,
                 buffer_freq,
                 train_freq,
                 reward_scaling,
                 policy_freq_high,
                 policy_freq_low,
                 gpu) -> None:
        """
        Constructor for the HIRO agent.
        """
        # get scale for subgoal
        self.scale_high = subgoal_space.high * np.ones(subgoal_dim)
        self.scale_low = scale_low
        self.model_save_freq = model_save_freq

        # create replay buffers
        self.low_level_replay_buffer = LowerControllerReplayBuffer(buffer_size)
        self.high_level_replay_buffer = HigherControllerReplayBuffer(buffer_size)

        # higher td3 controller
        self.high_con = HigherController(
            state_dim=state_dim,
            goal_dim=goal_dim,
            action_dim=subgoal_dim,
            scale=self.scale_high,
            model_path=model_path,
            policy_freq=policy_freq_high,
            replay_buffer=self.high_level_replay_buffer,
            minibatch_size=batch_size,
            gpu=gpu
        )

        # lower td3 controller
        self.low_con = LowerController(
            state_dim=state_dim,
            goal_dim=subgoal_dim,
            action_dim=action_dim,
            scale=self.scale_low,
            model_path=model_path,
            policy_freq=policy_freq_low,
            replay_buffer=self.low_level_replay_buffer,
            minibatch_size=batch_size,
            gpu=gpu
        )

        self.buffer_freq = buffer_freq

        self.train_freq = train_freq
        self.reward_scaling = reward_scaling
        self.episode_subreward = 0
        self.sr = 0
        self.state_arr = []
        self.action_arr = []
        self.cumulative_reward = 0

        self.start_training_steps = start_training_steps

    def act_high_level(self, obs, goal, subgoal, step=0):
        """
        high level actor
        """
        n_sg = self._choose_subgoal(step, self.last_obs, subgoal, obs, goal)
        self.sr = self.low_reward(self.last_obs, subgoal, obs)
        # clip values
        n_sg = np.clip(n_sg, a_min=-self.scale_high, a_max=self.scale_high)
        return n_sg

    def act_low_level(self, obs, goal):
        """
        low level actor,
        conditioned on an observation and goal.
        """
        self.last_obs = obs
        # goal = self.sg
        self.last_action = self.low_con.policy(obs, goal)
        self.last_action = np.clip(self.last_action, a_min=-self.scale_low, a_max=self.scale_low)
        return self.last_action

    def observe(self, obs, goal, subgoal, reward, done, reset, global_step=0, start_training_steps=0):
        """
        after getting feedback from the environment, observe,
        and train both the low and high level controllers.
        """

        if global_step >= start_training_steps:
            # start training once the global step surpasses
            # the start training steps
            self.low_con.observe(obs, subgoal, self.sr, done)

            if global_step % self.train_freq == 0 and len(self.action_arr) == self.train_freq:
                # train high level controller every self.train_freq steps
                self.high_con.agent.update_high_level_last_results(self.last_high_level_obs, self.last_high_level_goal, self.last_high_level_action)
                self.high_con.observe(self.low_con, self.state_arr, self.action_arr, self.cumulative_reward, goal, obs, done)
                self.action_arr = []
                self.state_arr = []

                # reset last high level obs, goal, action
                self.last_high_level_obs = torch.FloatTensor(obs)
                self.last_high_level_goal = torch.FloatTensor(goal)
                self.last_high_level_action = subgoal
                self.cumulative_reward = 0

            elif global_step % self.train_freq == 0:
                self.last_high_level_obs = torch.FloatTensor(obs)
                self.last_high_level_goal = torch.FloatTensor(goal)
                self.last_high_level_action = subgoal

            self.action_arr.append(self.last_action)
            self.state_arr.append(self.last_obs)
            self.cumulative_reward += (self.reward_scaling * reward)

    def _choose_subgoal(self, step, s, sg, n_s, goal):
        """
        chooses the next subgoal for the low level controller.
        """
        if step % self.buffer_freq == 0:
            sg = self.high_con.policy(s, goal)
        else:
            sg = self.subgoal_transition(s, sg, n_s)

        return sg

    def subgoal_transition(self, s, sg, n_s):
        """
        subgoal transition function, provided as input to the low
        level controller.
        """
        return s[:sg.shape[0]] + sg - n_s[:sg.shape[0]]

    def low_reward(self, s, sg, n_s):
        """
        reward function for low level controller.
        """
        abs_s = s[:sg.shape[0]] + sg
        return -np.sqrt(np.sum((abs_s - n_s[:sg.shape[0]])**2))

    def end_step(self):
        """
        ends a step within an episode.
        """
        self.episode_subreward += self.sr
        self.sg = self.n_sg

    def end_episode(self, episode, logger=None):
        """
        ends a full episode.
        """
        if logger:
            # log
            logger.write('reward/Intrinsic Reward', self.episode_subreward, episode)

            # Save Model
            if _is_update(episode, self.model_save_freq):
                self.save(episode=episode)

        self.episode_subreward = 0
        self.sr = 0

    def save(self, episode):
        """
        saves the model, aka the lower and higher controllers' parameters.
        """
        low_controller_dir = f'models/low_controller/episode_{episode}'
        high_controller_dir = f'models/high_controller/episode_{episode}'

        os.makedirs(low_controller_dir, exist_ok=True)
        os.makedirs(high_controller_dir, exist_ok=True)

        self.low_con.save(low_controller_dir)
        self.high_con.save(high_controller_dir)

    def load(self, episode):
        """
        loads from an episode.
        """
        low_controller_dir = f'models/low_controller/episode_{episode}'
        high_controller_dir = f'models/high_controller/episode_{episode}'
        try:
            self.low_con.load(low_controller_dir)
            self.high_con.load(high_controller_dir)
        except Exception as e:
            raise NotADirectoryError("Directory for loading internal state not found!")

    def set_to_train_(self):
        """
        sets an agent to train - this
        will make for a non-deterministic policy.
        """
        self.low_con.agent.training = True
        self.high_con.agent.training = True

    def set_to_eval_(self):
        """
        sets an agent to eval - making
        for the deterministic policy of td3
        """
        self.low_con.agent.training = False
        self.high_con.agent.training = False

    def get_statistics(self):
        """
        gets the statistics of all of the actors and critics for the high
        and low level controllers in the HIRO algorithm.
        """
        return [
            ("low_con_average_q1", _mean_or_nan(self.low_con.agent.q1_record)),
            ("low_con_average_q2", _mean_or_nan(self.low_con.agent.q2_record)),
            ("low_con_average_q_func1_loss", _mean_or_nan(self.low_con.agent.q_func1_loss_record)),
            ("low_con_average_q_func2_loss", _mean_or_nan(self.low_con.agent.q_func2_loss_record)),
            ("low_con_average_policy_loss", _mean_or_nan(self.low_con.agent.policy_loss_record)),
            ("low_con_policy_n_updates", self.low_con.agent.policy_n_updates),
            ("low_con_q_func_n_updates", self.low_con.agent.q_func_n_updates),

            ("high_con_average_q1", _mean_or_nan(self.high_con.agent.q1_record)),
            ("high_con_average_q2", _mean_or_nan(self.high_con.agent.q2_record)),
            ("high_con_average_q_func1_loss", _mean_or_nan(self.high_con.agent.q_func1_loss_record)),
            ("high_con_average_q_func2_loss", _mean_or_nan(self.high_con.agent.q_func2_loss_record)),
            ("high_con_average_policy_loss", _mean_or_nan(self.high_con.agent.policy_loss_record)),
            ("high_con_policy_n_updates", self.high_con.agent.policy_n_updates),
            ("high_con_q_func_n_updates", self.high_con.agent.q_func_n_updates),
        ]