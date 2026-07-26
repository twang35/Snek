from abc import ABCMeta

from Snake import *

from tf_agents.environments import py_environment
from tf_agents.specs import BoundedArraySpec
from tf_agents.trajectories.time_step import StepType
from tf_agents.trajectories.time_step import TimeStep
from tensorflow import convert_to_tensor


class SnakeEnvironment(py_environment.PyEnvironment, metaclass=ABCMeta):

    def __init__(self, discount=1.0, display=True, limit_fps=False):
        super().__init__()
        self._game = Game(display=display, limit_fps=limit_fps)
        self._discount = np.asarray(discount)
        self._observations = None
        self._total_steps = 0
        self.high_score = 0
        self.epsilon = 0.4

    def action_spec(self):
        # left, right, and forward
        return BoundedArraySpec((), np.int32, minimum=0, maximum=2, name='action')

    def observation_spec(self):
        food_obs = 6                # closer to, and lg(distance) to food
        body_and_wall_obs = 3       # body and wall is_collision
        # group obs are mixed by action
        head_with_tail_obs = 3      # head is in same group as tail
        head_with_food_obs = 0      # head is in same group as food
        total_groups_obs = 3        # lg(num_groups) for each action
        # end group obs
        perfect_game_move_obs = 3   # move results in a perfect game
        steps_until_starve_obs = 1  # steps until starve, capped at log2(500)
        remaining_spaces = 0        # open spaces left on grid
        game_over_obs = 1           # if game is over
        return BoundedArraySpec((food_obs
                                 + body_and_wall_obs
                                 + head_with_tail_obs
                                 + head_with_food_obs
                                 + total_groups_obs
                                 + perfect_game_move_obs
                                 + steps_until_starve_obs
                                 + remaining_spaces
                                 + game_over_obs,), np.float32)

    def _reset(self):
        self._game.reset()
        self._observations = self._game.get_observation()
        return self.to_tensor_time_step(StepType.FIRST, np.asarray(0.0), self._observations)

    def _step(self, action):
        if self._game.finished:
            return self.reset()

        self._total_steps += 1

        is_final, reward = self._game.step(TF_ACTION_TO_ACTIONS[action.item()])
        self._game.render()
        self._observations = self._game.get_observation()
        step_type = StepType.MID

        if is_final:
            step_type = StepType.LAST

        return self.to_tensor_time_step(step_type, reward, self._observations)

    def to_tensor_time_step(self, step_type, reward, observations):
        return TimeStep(step_type=convert_to_tensor(step_type, dtype=np.int32),
                        reward=convert_to_tensor(reward, dtype=np.float32),
                        discount=convert_to_tensor(self._discount, dtype=np.float32),
                        observation=convert_to_tensor(observations, dtype=np.float32))

    # Used to get the updated epsilon in the agent
    def get_updated_epsilon(self):
        if self.high_score < self._game.current_score:
            self.high_score = self._game.current_score
            print('new high score!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! ', self.high_score)

        return self.epsilon
