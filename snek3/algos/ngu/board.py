"""The board tensor the intrinsic modules read (`plans/algoExploration/f-ngu-agent57.md` §0, review 3): the
frame's analogue, decoded from `VecSnake.board_codes`' one-byte-per-cell codes into four channels over the
10 x 10 play area -- body, head, food, and body age (tail low, head 1) -- flattened to 400 values.

Why not the observation: the policy's observation carries the last eight moves, and the first pair of that
history **is** the action an inverse-dynamics classifier is asked to predict, so an embedding on it collapses
onto two bits. The board carries where things are and nothing about what was just pressed.
"""

import torch

from vectorized.vec_env import BOARD_CELLS, HEAD_CODE, FOOD_CODE

CHANNELS = 4
BOARD_LEN = CHANNELS * BOARD_CELLS          # 400


def decode(codes):
    """`(..., BOARD_CELLS) uint8` -> `(..., BOARD_LEN) float32`: body, head, food, age channels, concatenated."""
    codes = torch.as_tensor(codes)
    c = codes.to(torch.float32)
    body = ((c >= 1.0) & (c <= float(HEAD_CODE))).to(torch.float32)
    head = (c == float(HEAD_CODE)).to(torch.float32)
    food = (c == float(FOOD_CODE)).to(torch.float32)
    age = body * (c / float(HEAD_CODE))
    return torch.cat([body, head, food, age], dim=-1)
