from tf_agents.policies import tf_policy
from tf_agents.trajectories import time_step as ts, policy_step
from tf_agents.typing import types

CLOSER_TO_FOOD_REWARD = 1
FORWARD_TO_FOOD_REWARD = 10
MAX_FOOD_DISTANCE_REWARD = 10

DEATH_REWARD = -10000

HEAD_WITH_TAIL_REWARD = 1000


class TheSchmidPolicy(tf_policy.TFPolicy):

    def __init__(self, time_step_spec: ts.TimeStep,
                 action_spec: types.NestedTensorSpec, *args, **kwargs):
        super(TheSchmidPolicy, self).__init__(time_step_spec, action_spec, *args,
                                              **kwargs)

    def _distribution(self, time_step, policy_state: types.NestedTensorSpec):
        observation = time_step.observation
        food_obs = observation[0:7]
        col_obs = observation[7:10]
        hwt_obs = observation[10:13]

        reward_dist = {'left': 0, 'right': 0, 'forward': 0}

        calculate_food_rewards(reward_dist, food_obs)
        calculate_collision_rewards(reward_dist, col_obs)
        calculate_hwt_rewards(reward_dist, hwt_obs)

        best_action = 'none'
        best_reward = DEATH_REWARD * 2
        for key, value in reward_dist:
            if value > best_reward:
                best_action = key

        return policy_step.PolicyStep(action=best_action, state=(), info=())


def calculate_food_rewards(rewards_dist, food_obs):
    rewards_dist['left'] = rewards_dist['left'] \
                           + (food_obs[0] * CLOSER_TO_FOOD_REWARD) \
                           + (MAX_FOOD_DISTANCE_REWARD - food_obs[1])

    rewards_dist['right'] = rewards_dist['right'] \
                            + (food_obs[2] * CLOSER_TO_FOOD_REWARD) \
                            + (MAX_FOOD_DISTANCE_REWARD - food_obs[3])

    rewards_dist['forward'] = rewards_dist['forward'] \
                              + (food_obs[4] * FORWARD_TO_FOOD_REWARD) \
                              + (MAX_FOOD_DISTANCE_REWARD - food_obs[5])


def calculate_collision_rewards(rewards_dist, col_obs):
    rewards_dist['left'] += col_obs[0] * DEATH_REWARD
    rewards_dist['right'] += col_obs[1] * DEATH_REWARD
    rewards_dist['forward'] += col_obs[2] * DEATH_REWARD


def calculate_hwt_rewards(rewards_dist, hwt_obs):
    rewards_dist['left'] += hwt_obs[0] * HEAD_WITH_TAIL_REWARD
    rewards_dist['right'] += hwt_obs[1] * HEAD_WITH_TAIL_REWARD
    rewards_dist['forward'] += hwt_obs[2] * HEAD_WITH_TAIL_REWARD
