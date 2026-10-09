# b49d-btrplaintrunk-seed4

step **1,000,000** · 1000 evals · trailing **94.67** · peak **94.68** @998,000 · sef **42.5** · best30 **94.0** @996,000

## Config

| | |
|---|---|
| adam_epsilon | 1.953125e-05 |
| algo | btr |
| batch_size | 256 |
| beta_anneal_steps | 1 |
| btr_blocks | 3 |
| btr_layer_norm | False |
| btr_residual | False |
| btr_spectral_norm | True |
| collect_envs | 64 |
| discount | 0.997 |
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
| epsilon_schedule | linear |
| eval_interval | 1000 |
| eval_queue | True |
| eval_queue_depth | 16 |
| eval_workers | 8 |
| fc_layers | (320,) |
| fork_branches | 1 |
| gradient_clipping | 10.0 |
| graph_eval_episodes | 100 |
| guided_fraction | 0.0 |
| init_from | None |
| initial_collect_steps | 200000 |
| initial_epsilon | 1.0 |
| is_beta | 0.2 |
| is_beta_final | 0.2 |
| is_normalization | batch_max |
| is_weights | True |
| learning_rate | 0.0001 |
| max_steps | 1000000 |
| min_checkpoint_score | 40.0 |
| min_epsilon | 0.01 |
| munchausen_alpha | 0.9 |
| munchausen_l0 | -1.0 |
| munchausen_tau | 0.03 |
| n_step_update | 3 |
| priority_exponent | 0.2 |
| rainbow_double | False |
| rainbow_dueling | True |
| rainbow_epsilon_decay | geometric |
| rainbow_epsilon_zero_at | 0.5 |
| rainbow_head | iqn |
| rainbow_munchausen_logpi | online |
| rainbow_noisy | True |
| rainbow_noisy_sigma | 0.5 |
| rainbow_prefill_epsilon | schedule |
| rainbow_stream_width | 512 |
| replay_buffer_max_length | 1048576 |
| replay_ratio | 0.015625 |
| reset_alpha | 0.5 |
| reset_anneal_gamma |  |
| reset_anneal_n_step |  |
| reset_anneal_steps | 10000 |
| reset_interval | 0 |
| reset_stop_after | 0 |
| seed | 4 |
| target_update_period | 500 |
| target_update_tau | 1.0 |
| torch_threads | 1 |

![b49d-btrplaintrunk-seed4](b49d-btrplaintrunk-seed4.png)

## Evals

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 1000 | 9.23 | 9.23 | 0.0 | 32.0 | 7.564 | 0.0 | 0.8776 |
| 2000 | 0.71 | 4.97 | 0.0 | 5.0 | 0.109 | 0.0 | 0.85027 |
| 3000 | 1.4 | 3.78 | 0.0 | 19.0 | 0.568 | 0.0 | 0.82381 |
| ... | ... | ... | ... | ... | ... | ... | ... |
| 989000 | 94.82 | 94.65 | 87.0 | 95.0 | 190.302 | 97.0 | 0.0 |
| 990000 | 94.97 | 94.66 | 92.0 | 95.0 | 192.552 | 99.0 | 0.0 |
| 991000 | 94.46 | 94.64 | 46.0 | 95.0 | 188.894 | 96.0 | 0.0 |
| 992000 | 94.4 | 94.64 | 46.0 | 95.0 | 187.813 | 95.0 | 0.0 |
| 993000 | 94.81 | 94.64 | 86.0 | 95.0 | 189.247 | 96.0 | 0.0 |
| 994000 | 94.82 | 94.64 | 90.0 | 95.0 | 187.18 | 94.0 | 0.0 |
| 995000 | 94.95 | 94.65 | 93.0 | 95.0 | 190.431 | 97.0 | 0.0 |
| 996000 | 94.81 | 94.67 | 84.0 | 95.0 | 189.267 | 96.0 | 0.0 |
| 997000 | 94.28 | 94.67 | 80.0 | 95.0 | 178.287 | 86.0 | 0.0 |
| 998000 | 94.68 | 94.68 | 87.0 | 95.0 | 183.913 | 91.0 | 0.0 |
| 999000 | 94.66 | 94.67 | 81.0 | 95.0 | 185.96 | 93.0 | 0.0 |
| 1000000 | 94.75 | 94.67 | 81.0 | 95.0 | 187.135 | 94.0 | 0.0 |
