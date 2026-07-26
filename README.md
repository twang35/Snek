# Snek
Different agents that play snake, named by Josh

## TheSchlort
Random movements to learn how to control the snek.

## TheSchmid
Human-coded agent that tries. ~20% perfect game rate.


![theSchmid](https://user-images.githubusercontent.com/5852883/214492458-7fef1c79-1abb-4907-a2bb-cb3ddd4fdd7a.gif)


## TheSchlong
Reinforcement learning agent. ~76% perfect game rate.


![theSchlong](https://user-images.githubusercontent.com/5852883/214492576-f493ec8a-afc0-4e86-9ed5-35e7c6b8be19.gif)

## Snek2

`snek2/` is the actively-developed reinforcement learning agent (a working
copy of TheSchlong). To start it:

```
conda activate snek
cd snek2
python snek2.py <policy_name>
```

The argument passed in is the policy name. It's used both as the checkpoint
directory under `snek2/savedPolicies/<policy_name>/` and as the results
window title, so different policy names can be trained independently without
overwriting each other's checkpoints.

To train multiple policies at once, open a separate terminal window for each
and run one per window:

```
# terminal 1
python snek2.py train1

# terminal 2
python snek2.py train2

# terminal 3
python snek2.py train3

# terminal 4
python snek2.py train4
```

### Run documentation

Every eval refreshes the live graph window and also writes three files to
`snek2/runs/`, named after the policy:

```
snek2/runs/train.png             the graph, same image as the window
snek2/runs/train.md              graph, config, and eval table
snek2/runs/train_history.json    graph data, so restarts continue the curve
```

The graph covers the policy's whole history rather than just the current run.
Stopping and restarting `train` picks the curve back up where it left off and
draws a dashed vertical line at the step where training resumed, so a dip after a
restart is easy to tell apart from the policy getting worse.

The `.md` config table is generated from the values the run actually used, so it
can't drift. `_history.json` is local state and is gitignored; the `.png` and
`.md` are the documentation and can be committed.

To run without any windows (over ssh, or to avoid the graph stealing focus):

```
SDL_VIDEODRIVER=dummy python snek2.py <policy_name>
```

