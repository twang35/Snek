import random

from Snake import Game as game
from time import sleep
import pygame
from pygame.locals import *

slowMode = True
numRuns = 5

def random_game():
    averageScore = 0
    for _ in range(numRuns):
        env = game()
        env.reset()
        action = -1

        max_moves = 69420

        for _ in range(max_moves):
            action = random.randrange(0, 3)

            if slowMode:
                sleep(0.1)
            env.render()

            done, score = env.step(action)

            if done:
                print('Score: ', score)
                averageScore += score
                break

    print('Average score: ', averageScore/numRuns)

if __name__ == "__main__":
    random_game()