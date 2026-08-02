from abc import ABCMeta

from Snake import *

from tf_agents.environments import py_environment
from tf_agents.specs import BoundedArraySpec
from tf_agents.trajectories.time_step import StepType
from tf_agents.trajectories.time_step import TimeStep
from tensorflow import convert_to_tensor


class SnakeEnvironment(py_environment.PyEnvironment, metaclass=ABCMeta):

    def __init__(self, discount=1.0, display=True, limit_fps=False, policy_name=''):
        super().__init__()
        self._game = Game(display=display, limit_fps=limit_fps, policy_name=policy_name)
        self._discount = np.asarray(discount)
        self._observations = None
        self._total_steps = 0
        self.high_score = 0

    def action_spec(self):
        # left, right, and forward
        return BoundedArraySpec((), np.int32, minimum=0, maximum=2, name='action')

    def observation_spec(self):
        food_obs = 6                # closer to, and lg(distance) to food
        body_and_wall_obs = 3       # body and wall is_collision
        # group obs are mixed by action
        head_with_tail_obs = 3      # head is in same group as tail
        # Switched on 2026-08-02, and stricter than the version this slot was reserved for:
        # head, food *and* tail in one group, so there is a way to the food and a way back out.
        head_with_food_obs = 3      # safe to chase the food
        total_groups_obs = 3        # lg(num_groups) for each action
        # end group obs
        perfect_game_move_obs = 3   # move results in a perfect game
        steps_until_starve_obs = 1  # starve budget left, lg-compressed and scaled to [0, 1]
        # Supersedes the disabled `remaining_spaces` slot that sat here: open cells are the
        # complement of snake length, so this is that signal, normalised, and switched on.
        snake_length_obs = 1        # fraction of the board the snake fills
        # `game_over` used to live here, and it was load-bearing for the wrong reason: terminal
        # steps carried a non-zero discount, so the loss bootstrapped off the terminal state's
        # Q-values and the only way the network could learn those are worth nothing was to key
        # on this flag. With the discount zeroed in to_tensor_time_step() the bootstrap is gone,
        # nothing reads a terminal state's value, and the flag was 0 in 100% of the states a
        # policy ever acts in — a constant input.
        return BoundedArraySpec((food_obs
                                 + body_and_wall_obs
                                 + head_with_tail_obs
                                 + head_with_food_obs
                                 + total_groups_obs
                                 + perfect_game_move_obs
                                 + steps_until_starve_obs
                                 + snake_length_obs,), np.float32)

    def set_display(self, enabled):
        self._game.set_display(enabled)

    def get_score(self):
        return self._game.current_score

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

        if self.high_score < self._game.current_score:
            self.high_score = self._game.current_score
            if snake_constants.DEBUG_LOGGING:
                print('new high score!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! ',
                      self.high_score)

        if is_final:
            step_type = StepType.LAST

        return self.to_tensor_time_step(step_type, reward, self._observations)

    def to_tensor_time_step(self, step_type, reward, observations):
        # A terminal step must carry discount 0, because that zero is the *only* thing that stops
        # the bootstrap. tf_agents' own ts.termination() sets it; this environment did not, and
        # nothing else compensates: DdqnAgent is built without `gamma`, so gamma defaults to 1.0,
        # and dqn_agent._loss computes `discounts = gamma * next_time_steps.discount` and then
        # `td_targets = rewards + discounts * next_q_values`. The valid_mask there only drops
        # transitions whose *current* step is LAST, not the bootstrap off a terminal next state.
        #
        # So every episode's final transition was trained toward `reward + 0.9975 * V(terminal)`
        # rather than toward `reward`, which quietly discounted the -5 death penalty by whatever
        # value the network assigned the terminal observation.
        #
        # It also matters for n-step returns, where the composed reward is
        # `r_t + g*d_t*r_{t+1} + g^2*d_t*d_{t+1}*r_{t+2} + ...` — those per-step d values are the
        # only truncation at an episode boundary, so a non-zero one sums rewards straight through
        # the end of the episode.
        discount = 0.0 if step_type == StepType.LAST else self._discount
        return TimeStep(step_type=convert_to_tensor(step_type, dtype=np.int32),
                        reward=convert_to_tensor(reward, dtype=np.float32),
                        discount=convert_to_tensor(discount, dtype=np.float32),
                        observation=convert_to_tensor(observations, dtype=np.float32))
