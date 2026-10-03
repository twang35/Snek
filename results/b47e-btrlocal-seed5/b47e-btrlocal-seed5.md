# b47e-btrlocal-seed5

step **1,000,000** · 1000 evals · trailing **94.38** · peak **94.55** @951,000 · sef **88.9** · best30 **96.7** @941,000

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
| seed | 5 |
| target_update_period | 8 |
| target_update_tau | 1.0 |
| torch_threads | 1 |

![b47e-btrlocal-seed5](b47e-btrlocal-seed5.png)

## Evals

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 1000 | 1.16 | 1.16 | 1.0 | 3.0 | 0.202 | 0.0 | 0.4 |
| 2000 | 2.16 | 1.66 | 1.0 | 4.0 | -0.492 | 0.0 | 0.4 |
| 3000 | 9.61 | 4.31 | 2.0 | 28.0 | 4.603 | 0.0 | 0.4 |
| ... | ... | ... | ... | ... | ... | ... | ... |
| 989000 | 94.02 | 94.45 | 25.0 | 95.0 | 189.637 | 97.0 | 0.002 |
| 990000 | 94.98 | 94.46 | 93.0 | 95.0 | 192.607 | 99.0 | 0.002 |
| 991000 | 94.8 | 94.46 | 82.0 | 95.0 | 189.406 | 96.0 | 0.002 |
| 992000 | 93.89 | 94.43 | 33.0 | 95.0 | 188.554 | 96.0 | 0.002 |
| 993000 | 94.82 | 94.43 | 88.0 | 95.0 | 189.324 | 96.0 | 0.002 |
| 994000 | 94.51 | 94.41 | 56.0 | 95.0 | 189.106 | 96.0 | 0.002 |
| 995000 | 93.75 | 94.39 | 29.0 | 95.0 | 184.327 | 92.0 | 0.002 |
| 996000 | 94.38 | 94.41 | 46.0 | 95.0 | 189.995 | 97.0 | 0.002 |
| 997000 | 94.89 | 94.42 | 89.0 | 95.0 | 189.445 | 96.0 | 0.002 |
| 998000 | 94.74 | 94.44 | 84.0 | 95.0 | 189.335 | 96.0 | 0.002 |
| 999000 | 93.03 | 94.38 | 17.0 | 95.0 | 187.692 | 96.0 | 0.002 |
| 1000000 | 94.83 | 94.38 | 85.0 | 95.0 | 191.431 | 98.0 | 0.002 |
