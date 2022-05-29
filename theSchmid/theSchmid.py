from queue import PriorityQueue
from time import sleep
from observationHelpers import *


# slow_mode = True
slow_mode = False
time_delay = 0.2
num_runs = 10
max_moves = 69000


def run_game():
    average_score = 0
    average_steps = 0
    for _ in range(num_runs):
        env = Game()
        env.reset()
        chosen_action = -1

        for step in range(max_moves):
            pygame.event.get()
            if slow_mode:
                sleep(time_delay)
            kitchen_sink = env.render()

            if chosen_action == -1:
                # select random first action because grid is not yet rendered
                chosen_action = ('', (DIRECTIONS[random.randrange(0, 3)], False))
            else:
                action_scores = PriorityQueue()
                for action in DIRECTIONS:
                    score, to_check = calculate_score(action, kitchen_sink)
                    action_scores.put((-1*score, (action, to_check)))
                chosen_action = action_scores.get()

            if chosen_action[1][1]:
                print('its happening!')

            kitchen_sink = env.step(chosen_action[1][0])

            if kitchen_sink.game_over or max_moves-1 == step:
                print('Score: ', kitchen_sink.current_score, ' steps: ', step)
                average_score += kitchen_sink.current_score
                average_steps += step
                break

    print('Average score: ', average_score / num_runs)
    print('Average steps: ', average_steps / num_runs)


if __name__ == "__main__":
    run_game()
