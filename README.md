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

