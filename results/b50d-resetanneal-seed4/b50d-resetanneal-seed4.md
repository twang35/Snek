# b50d-resetanneal-seed4

step **3,000,000** · 3000 evals · trailing **93.57** · peak **93.59** @2,931,000 · sef **0.0** · best30 **60.8** @2,798,000

## Config

| | |
|---|---|
| adam_epsilon | 1e-07 |
| algo | qrdqn |
| batch_size | 128 |
| beta_anneal_steps | 300000 |
| collect_envs | 1 |
| discount | 0.99 |
| dist_atoms | 51 |
| dist_embedding | 64 |
| dist_fraction_entropy | 0.001 |
| dist_fraction_lr | 2.5e-09 |
| dist_kappa | 1.0 |
| dist_policy_samples | 32 |
| dist_quantiles | 32 |
| dist_risk_alpha | 1.0 |
| dist_risk_train | False |
| dist_tau_prime_samples | 64 |
| dist_tau_samples | 64 |
| dist_v_max | 110.0 |
| dist_v_min | -10.0 |
| epsilon_anneal_steps | 250000 |
| epsilon_schedule | eval |
| eval_interval | 1000 |
| eval_queue | True |
| eval_queue_depth | 16 |
| eval_workers | 8 |
| fc_layers | (320,) |
| fork_branches | 4 |
| fork_max_steps | 60 |
| fork_min_length | 85 |
| fork_prob | 0.5 |
| gradient_clipping | 0.0 |
| graph_eval_episodes | 100 |
| guided_fraction | 0.8 |
| init_from | None |
| initial_collect_steps | 2000 |
| initial_epsilon | 0.4 |
| is_beta | 0.4 |
| is_beta_final | 1.0 |
| is_weights | True |
| learning_rate | 1e-05 |
| max_steps | 3000000 |
| min_checkpoint_score | 40.0 |
| min_epsilon | 0.002 |
| munchausen_alpha | 0.9 |
| munchausen_l0 | -1.0 |
| munchausen_tau | 0.03 |
| n_step_update | 1 |
| priority_exponent | 0.6 |
| replay_buffer_max_length | 100000 |
| replay_ratio | 1.0 |
| reset_alpha | 0.5 |
| reset_anneal_gamma | 0.97,0.997 |
| reset_anneal_n_step | 10,3 |
| reset_anneal_steps | 10000 |
| reset_interval | 600000 |
| reset_stop_after | 10500000 |
| seed | 4 |
| target_update_period | 8 |
| target_update_tau | 1.0 |
| torch_threads | 1 |

![b50d-resetanneal-seed4](b50d-resetanneal-seed4.png)

## Evals

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 1000 | 2.65 | 2.65 | 0.0 | 18.0 | 1.687 | 0.0 | 0.4 |
| 2000 | 6.95 | 4.8 | 0.0 | 15.0 | 2.342 | 0.0 | 0.4 |
| 3000 | 10.67 | 6.76 | 0.0 | 34.0 | 5.968 | 0.0 | 0.4 |
| ... | ... | ... | ... | ... | ... | ... | ... |
| 2989000 | 92.54 | 93.55 | 2.0 | 95.0 | 149.81 | 60.0 | 0.00321 |
| 2990000 | 93.68 | 93.56 | 86.0 | 95.0 | 142.823 | 52.0 | 0.00321 |
| 2991000 | 93.83 | 93.57 | 85.0 | 95.0 | 150.032 | 59.0 | 0.00321 |
| 2992000 | 93.71 | 93.57 | 74.0 | 95.0 | 149.086 | 58.0 | 0.0032 |
| 2993000 | 93.58 | 93.57 | 82.0 | 95.0 | 145.91 | 55.0 | 0.00321 |
| 2994000 | 93.7 | 93.57 | 84.0 | 95.0 | 154.384 | 63.0 | 0.00322 |
| 2995000 | 93.58 | 93.58 | 84.0 | 95.0 | 145.056 | 54.0 | 0.00322 |
| 2996000 | 93.72 | 93.58 | 82.0 | 95.0 | 154.312 | 63.0 | 0.00324 |
| 2997000 | 93.79 | 93.59 | 82.0 | 95.0 | 149.055 | 58.0 | 0.00323 |
| 2998000 | 93.46 | 93.56 | 82.0 | 95.0 | 140.54 | 50.0 | 0.00325 |
| 2999000 | 93.55 | 93.55 | 82.0 | 95.0 | 145.857 | 55.0 | 0.00326 |
| 3000000 | 93.65 | 93.57 | 76.0 | 95.0 | 151.026 | 60.0 | 0.00326 |
