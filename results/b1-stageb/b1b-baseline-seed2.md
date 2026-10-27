# b1b-baseline-seed2

step **3,000,000** · 3000 evals · trailing **92.65** · peak **93.21** @1,217,000 · sef **0.0** · best30 **58.3** @1,226,000

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
| seed | 2 |
| target_update_period | 8 |
| target_update_tau | 1.0 |
| torch_threads | 1 |

![b1b-baseline-seed2](b1b-baseline-seed2.png)

## Evals

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 1000 | 0.74 | 0.74 | 0.0 | 3.0 | 0.187 | 0.0 | 0.4 |
| 2000 | 4.01 | 2.38 | 0.0 | 15.0 | 3.445 | 0.0 | 0.4 |
| 3000 | 9.87 | 4.87 | 1.0 | 95.0 | 10.233 | 1.0 | 0.2 |
| ... | ... | ... | ... | ... | ... | ... | ... |
| 2989000 | 92.12 | 92.75 | 31.0 | 95.0 | 133.321 | 43.0 | 0.00428 |
| 2990000 | 92.06 | 92.73 | 56.0 | 95.0 | 126.019 | 36.0 | 0.0043 |
| 2991000 | 93.18 | 92.74 | 80.0 | 95.0 | 140.252 | 49.0 | 0.00429 |
| 2992000 | 92.24 | 92.7 | 10.0 | 95.0 | 138.305 | 48.0 | 0.0043 |
| 2993000 | 92.39 | 92.67 | 41.0 | 95.0 | 137.569 | 47.0 | 0.00431 |
| 2994000 | 93.8 | 92.73 | 86.0 | 95.0 | 147.049 | 55.0 | 0.00427 |
| 2995000 | 92.69 | 92.74 | 12.0 | 95.0 | 137.776 | 47.0 | 0.00426 |
| 2996000 | 92.36 | 92.72 | 59.0 | 95.0 | 141.59 | 51.0 | 0.00424 |
| 2997000 | 92.23 | 92.71 | 8.0 | 95.0 | 123.099 | 33.0 | 0.00423 |
| 2998000 | 91.43 | 92.68 | 21.0 | 95.0 | 130.656 | 41.0 | 0.00428 |
| 2999000 | 92.05 | 92.63 | 47.0 | 95.0 | 133.243 | 43.0 | 0.00431 |
| 3000000 | 92.45 | 92.65 | 25.0 | 95.0 | 141.761 | 51.0 | 0.00432 |
