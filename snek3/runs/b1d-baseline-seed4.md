# b1d-baseline-seed4

step **10,000** · 10 evals · trailing **41.99** · peak **41.99** @10,000 · sef **0.0** · best30 **0.0** @10,000

## Config

| | |
|---|---|
| adam_epsilon | 1e-07 |
| algo | dqn |
| batch_size | 128 |
| beta_anneal_steps | 300000 |
| collect_envs | 1 |
| discount | 0.99 |
| eval_interval | 1000 |
| fc_layers | (320,) |
| fork_branches | 4 |
| fork_max_steps | 60 |
| fork_min_length | 85 |
| fork_prob | 0.5 |
| gradient_clipping | 0.0 |
| graph_eval_episodes | 100 |
| guided_fraction | 0.8 |
| initial_collect_steps | 2000 |
| initial_epsilon | 0.4 |
| is_beta | 0.4 |
| is_beta_final | 1.0 |
| is_weights | True |
| learning_rate | 1e-05 |
| max_steps | 3000000 |
| min_checkpoint_score | 40.0 |
| min_epsilon | 0.002 |
| n_step_update | 1 |
| priority_exponent | 0.6 |
| replay_buffer_max_length | 100000 |
| replay_ratio | 1.0 |
| seed | 4 |
| target_update_period | 8 |
| target_update_tau | 1.0 |
| torch_threads | 1 |

![b1d-baseline-seed4](b1d-baseline-seed4.png)

## Evals

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 1000 | 2.24 | 2.24 | 0.0 | 12.0 | 1.667 | 0.0 | 0.4 |
| 2000 | 2.63 | 2.44 | 0.0 | 12.0 | 2.057 | 0.0 | 0.4 |
| 3000 | 4.83 | 3.23 | 1.0 | 21.0 | 4.234 | 0.0 | 0.2 |
| 4000 | 19.37 | 7.27 | 1.0 | 95.0 | 19.587 | 1.0 | 0.1 |
| 5000 | 70.34 | 19.88 | 1.0 | 93.0 | 68.755 | 0.0 | 0.025 |
| 6000 | 70.02 | 28.24 | 23.0 | 87.0 | 68.773 | 0.0 | 0.01245 |
| 7000 | 69.32 | 34.11 | 6.0 | 87.0 | 68.083 | 0.0 | 0.01246 |
| 8000 | 65.15 | 37.99 | 3.0 | 90.0 | 63.944 | 0.0 | 0.01246 |
| 9000 | 57.08 | 40.11 | 3.0 | 87.0 | 55.99 | 0.0 | 0.01247 |
| 10000 | 58.92 | 41.99 | 1.0 | 90.0 | 57.78 | 0.0 | 0.01247 |
