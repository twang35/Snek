# b47g-btrlocal-seed7

step **1,000,000** · 1000 evals · trailing **94.1** · peak **94.53** @483,000 · sef **87.4** · best30 **96.4** @503,000

## Config

| | |
|---|---|
| adam_epsilon | 1e-07 |
| algo | btr |
| batch_size | 128 |
| beta_anneal_steps | 300000 |
| btr_blocks | 3 |
| btr_layer_norm | False |
| btr_residual | True |
| btr_spectral_norm | True |
| collect_envs | 1 |
| discount | 0.99 |
| dist_atoms | 51 |
| dist_embedding | 64 |
| dist_kappa | 1.0 |
| dist_policy_samples | 8 |
| dist_quantiles | 32 |
| dist_tau_prime_samples | 8 |
| dist_tau_samples | 8 |
| dist_v_max | 110.0 |
| dist_v_min | -10.0 |
| epsilon_anneal_steps | 2000000 |
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
| is_normalization | mean |
| is_weights | True |
| learning_rate | 1e-05 |
| max_steps | 1000000 |
| min_checkpoint_score | 40.0 |
| min_epsilon | 0.002 |
| munchausen_alpha | 0.9 |
| munchausen_l0 | -1.0 |
| munchausen_tau | 0.03 |
| n_step_update | 3 |
| priority_exponent | 0.6 |
| rainbow_double | False |
| rainbow_dueling | True |
| rainbow_epsilon_decay | geometric |
| rainbow_epsilon_zero_at | 0.0 |
| rainbow_head | quantile |
| rainbow_munchausen_logpi | online |
| rainbow_noisy | True |
| rainbow_noisy_sigma | 0.5 |
| rainbow_prefill_epsilon | schedule |
| rainbow_stream_width | 512 |
| replay_buffer_max_length | 100000 |
| replay_ratio | 1.0 |
| reset_alpha | 0.5 |
| reset_anneal_gamma |  |
| reset_anneal_n_step |  |
| reset_anneal_steps | 10000 |
| reset_interval | 0 |
| reset_stop_after | 0 |
| seed | 7 |
| target_update_period | 8 |
| target_update_tau | 1.0 |
| torch_threads | 1 |

![b47g-btrlocal-seed7](b47g-btrlocal-seed7.png)

## Evals

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 1000 | 6.42 | 6.42 | 2.0 | 15.0 | 1.412 | 0.0 | 0.4 |
| 2000 | 3.76 | 5.09 | 2.0 | 10.0 | -1.205 | 0.0 | 0.4 |
| 3000 | 6.72 | 5.63 | 2.0 | 18.0 | 1.713 | 0.0 | 0.4 |
| ... | ... | ... | ... | ... | ... | ... | ... |
| 989000 | 94.41 | 94.22 | 58.0 | 95.0 | 184.909 | 92.0 | 0.002 |
| 990000 | 94.32 | 94.2 | 60.0 | 95.0 | 183.868 | 91.0 | 0.002 |
| 991000 | 94.28 | 94.18 | 43.0 | 95.0 | 188.909 | 96.0 | 0.002 |
| 992000 | 94.76 | 94.18 | 78.0 | 95.0 | 190.399 | 97.0 | 0.002 |
| 993000 | 93.81 | 94.16 | 1.0 | 95.0 | 188.397 | 96.0 | 0.002 |
| 994000 | 93.64 | 94.14 | 58.0 | 95.0 | 182.174 | 90.0 | 0.002 |
| 995000 | 94.63 | 94.13 | 78.0 | 95.0 | 185.13 | 92.0 | 0.002 |
| 996000 | 93.96 | 94.14 | 21.0 | 95.0 | 185.551 | 93.0 | 0.002 |
| 997000 | 92.94 | 94.08 | 6.0 | 95.0 | 185.592 | 94.0 | 0.002 |
| 998000 | 94.71 | 94.09 | 86.0 | 95.0 | 188.328 | 95.0 | 0.002 |
| 999000 | 94.35 | 94.11 | 44.0 | 95.0 | 188.928 | 96.0 | 0.002 |
| 1000000 | 94.31 | 94.1 | 70.0 | 95.0 | 183.911 | 91.0 | 0.002 |
