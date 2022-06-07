from abc import ABCMeta

from Snake import *

from tf_agents.environments import py_environment
from tf_agents.specs import BoundedArraySpec
from tf_agents.specs import ArraySpec
from tf_agents.trajectories.time_step import StepType
from tf_agents.trajectories.time_step import TimeStep
from tensorflow import convert_to_tensor


class SnakeEnvironment(py_environment.PyEnvironment, metaclass=ABCMeta):
    ACTIONS = {0: 'left',
               1: 'right',
               2: 'up',
               3: 'down'}

    def __init__(self, discount=1.0, display=True, limit_fps=False):
        super().__init__()
        self._game = Game(display=display, limit_fps=limit_fps)
        self._discount = np.asarray(discount)
        self._observations = None
        self._total_steps = 0
        self.high_score = 0

    def action_spec(self):
        return BoundedArraySpec((), np.int32, minimum=0, maximum=3, name='action')

    def observation_spec(self):
        food_obs = 12               # closer to, on top of food, distance to food
        body_and_wall_obs = 4       # body and wall is_collision
        head_with_tail_obs = 4      # head is in same group as tail
        steps_until_starve_obs = 1  # only steps until starve
        return BoundedArraySpec((food_obs
                                 + body_and_wall_obs
                                 + head_with_tail_obs
                                 + steps_until_starve_obs,), np.float32)

    def _reset(self):
        self._game.reset()
        self._observations = self._game.get_observation()
        return self.to_tensor_time_step(StepType.FIRST, np.asarray(0.0), self._observations)

    def _step(self, action):
        if self._game.finished:
            return self.reset()

        self._total_steps += 1

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

    # Used to update the epsilon in the agent
    def get_updated_epsilon(self):
        if self.high_score < self._game.current_score:
            self.high_score = self._game.current_score
            print('new high score!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! ', self.high_score)
        if self.high_score < 10:
            return 0.5
        elif self.high_score < 20:
            return 0.3
        else:
            return 0.1

