from abc import ABCMeta

from Snake import *

from tf_agents.environments import py_environment
from tf_agents.specs import BoundedArraySpec
from tf_agents.specs import ArraySpec
from tf_agents.trajectories.time_step import StepType
from tf_agents.trajectories.time_step import TimeStep
from tensorflow import convert_to_tensor


class SnakeEnvironment(py_environment.PyEnvironment, metaclass=ABCMeta):
    ACTIONS = {0: 'right',
               1: 'left',
               2: 'up',
               3: 'down'}

    def __init__(self, discount=1.0, display=True):
        self._game = Game(display=display)
        self._discount = np.asarray(discount)
        self._observations = None

    def action_spec(self):
        return BoundedArraySpec((), np.int32, minimum=0, maximum=3, name='action')

    def observation_spec(self):
        food_obs = 8                # closer to and on top of food in each direction
        body_and_wall_obs = 4       # body and wall collisions
        return BoundedArraySpec((food_obs
                                 + body_and_wall_obs,), np.float32)

    def _reset(self):
        self._game.reset()
        self._observations = self._game.get_observation()
        return self.to_tensor_time_step(StepType.FIRST, np.asarray(0.0), self._observations)

    def _step(self, action):
        if self._game.finished:
            return self.reset()

        is_final, reward = self._game.step(SnakeEnvironment.ACTIONS[action.item()])
        self._observations = self._game.get_observation()
        self._game.render()
        step_type = StepType.MID

        if is_final:
            step_type = StepType.LAST

        return self.to_tensor_time_step(step_type, reward, self._observations)

    def to_tensor_time_step(self, step_type, reward, observations):
        return TimeStep(step_type=convert_to_tensor(step_type, dtype=np.int32),
                        reward=convert_to_tensor(reward, dtype=np.float32),
                        discount=convert_to_tensor(self._discount, dtype=np.float32),
                        observation=convert_to_tensor(observations, dtype=np.float32))
