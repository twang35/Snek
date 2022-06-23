This version of theSchlong operates in the cardinal North, East, South, West directions. The regular game of snake generally operates with such controls, so it made sense to start this way.

However, this is not great because it takes 4 times as long to learn anything because each direction is essentially a new state for Snek. Each direction behaves differently for edge food, corner, food, distanced food, etc. and training takes too long and is inconsistent.

Even when one direction learns something perfectly, another direction can cause a death because it hasn't learned yet.

The next version of theSchlong will be relative direction with only left, right, and forward directions. Though mapping will need to be done between observation and action selection, this should still be faster with much more consistent training.
