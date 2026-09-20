# b38e-iqncvar25-seed5

step **919,000** · 919 evals · trailing **92.32** · peak **93.0** @355,000 · sef **0.0** · best30 **47.1** @210,000

## Config

| | |
|---|---|
| adam_epsilon | 1e-07 |
| algo | iqn |
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
| dist_risk_alpha | 0.25 |
| dist_risk_train | True |
| dist_tau_prime_samples | 8 |
| dist_tau_samples | 8 |
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
| max_steps | 2000000 |
| min_checkpoint_score | 40.0 |
| min_epsilon | 0.002 |
| munchausen_alpha | 0.0 |
| munchausen_l0 | -1.0 |
| munchausen_tau | 0.03 |
| n_step_update | 1 |
| priority_exponent | 0.6 |
| replay_buffer_max_length | 100000 |
| replay_ratio | 1.0 |
| seed | 5 |
| target_update_period | 8 |
| target_update_tau | 1.0 |
| torch_threads | 1 |

![b38e-iqncvar25-seed5](b38e-iqncvar25-seed5.png)

## Evals

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 1000 | 0.86 | 0.86 | 0.0 | 4.0 | 0.301 | 0.0 | 0.4 |
| 2000 | 0.49 | 0.68 | 0.0 | 5.0 | -0.063 | 0.0 | 0.4 |
| 3000 | 0.62 | 0.66 | 0.0 | 4.0 | 0.067 | 0.0 | 0.4 |
| ... | ... | ... | ... | ... | ... | ... | ... |
| 908000 | 92.78 | 92.37 | 78.0 | 95.0 | 124.824 | 35.0 | 0.0059 |
| 909000 | 92.52 | 92.37 | 85.0 | 95.0 | 118.829 | 29.0 | 0.00594 |
| 910000 | 91.43 | 92.34 | 54.0 | 95.0 | 107.937 | 20.0 | 0.00592 |
| 911000 | 92.08 | 92.37 | 70.0 | 95.0 | 123.99 | 35.0 | 0.00596 |
| 912000 | 90.83 | 92.31 | 18.0 | 95.0 | 120.917 | 33.0 | 0.00602 |
| 913000 | 92.32 | 92.3 | 64.0 | 95.0 | 124.167 | 35.0 | 0.00602 |
| 914000 | 92.74 | 92.3 | 78.0 | 95.0 | 125.766 | 36.0 | 0.00602 |
| 915000 | 92.7 | 92.3 | 76.0 | 95.0 | 126.707 | 37.0 | 0.00604 |
| 916000 | 92.54 | 92.33 | 70.0 | 95.0 | 123.652 | 34.0 | 0.00604 |
| 917000 | 91.72 | 92.31 | 11.0 | 95.0 | 121.88 | 33.0 | 0.00604 |
| 918000 | 92.73 | 92.33 | 24.0 | 95.0 | 128.775 | 39.0 | 0.00601 |
| 919000 | 92.58 | 92.32 | 78.0 | 95.0 | 119.325 | 30.0 | 0.006 |
