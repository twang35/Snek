# b47g-btrlocal-seed7

step **919,000** · 919 evals · trailing **93.98** · peak **94.53** @483,000 · sef **86.3** · best30 **96.4** @503,000

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
| 908000 | 94.1 | 93.57 | 53.0 | 95.0 | 188.692 | 96.0 | 0.002 |
| 909000 | 94.55 | 93.59 | 74.0 | 95.0 | 187.096 | 94.0 | 0.002 |
| 910000 | 94.44 | 93.61 | 70.0 | 95.0 | 188.06 | 95.0 | 0.002 |
| 911000 | 94.5 | 93.66 | 47.0 | 95.0 | 190.039 | 97.0 | 0.002 |
| 912000 | 94.58 | 93.67 | 64.0 | 95.0 | 189.126 | 96.0 | 0.002 |
| 913000 | 94.42 | 93.7 | 72.0 | 95.0 | 184.819 | 92.0 | 0.002 |
| 914000 | 94.25 | 93.78 | 49.0 | 95.0 | 185.801 | 93.0 | 0.002 |
| 915000 | 93.19 | 93.84 | 45.0 | 95.0 | 180.664 | 89.0 | 0.002 |
| 916000 | 93.08 | 93.84 | 2.0 | 95.0 | 183.578 | 92.0 | 0.002 |
| 917000 | 94.7 | 93.89 | 88.0 | 95.0 | 187.294 | 94.0 | 0.002 |
| 918000 | 94.81 | 93.9 | 78.0 | 95.0 | 191.437 | 98.0 | 0.002 |
| 919000 | 94.84 | 93.98 | 79.0 | 95.0 | 192.465 | 99.0 | 0.002 |
