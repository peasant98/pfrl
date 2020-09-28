"""
Hindsight Experience Replay.
"""


class HERModule():
    def __init__(self):
        pass


class GoalConditionedHERModule():
    """
    explicitly goal conditioned (concatenation is taken care of elsewhere)
    HER.
    """
    def __init__(self, goal_sampling_fnc, recompute_reward_fnc, replay_buffer):
        self.goal_sampling_fnc = goal_sampling_fnc
        self.recompute_reward_fnc = recompute_reward_fnc
        self.replay_buffer = replay_buffer

    def append(self, episode_trajectory):
        """
        given the current episode, appends
        to the goal conditioned HER replay buffer.

        Arguments
        ---------
        """
        for transition in episode_trajectory:
            action = transition['action']
            state = transition['state']
            # these new goals should be from the end of episode
            new_goals_set = self.goal_sampling_fnc()
            for new_goal in new_goals_set:
                # recompute reward with new goal
                new_r = self.recompute_reward_fnc(state, new_goal, action)
                # store a new transition in the replay buffer
                self.replay_buffer.append()


class HigherLevelGoalConditionedHERModule():
    def __init__(self):
        pass
