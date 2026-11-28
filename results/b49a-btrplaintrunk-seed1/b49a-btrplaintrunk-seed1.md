# b49a-btrplaintrunk-seed1

step **1,000,000** · 1000 evals · trailing **94.65** · peak **94.7** @976,000 · sef **42.6** · best30 **96.0** @941,000

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
| seed | 1 |
| target_update_period | 500 |
| target_update_tau | 1.0 |
| torch_threads | 1 |

![b49a-btrplaintrunk-seed1](b49a-btrplaintrunk-seed1.png)

## Evals

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 1000 | 11.54 | 11.54 | 0.0 | 27.0 | 8.534 | 0.0 | 0.87759 |
| 2000 | 3.6 | 7.57 | 0.0 | 13.0 | 2.69 | 0.0 | 0.85027 |
| 3000 | 10.04 | 8.39 | 0.0 | 22.0 | 5.591 | 0.0 | 0.82381 |
| ... | ... | ... | ... | ... | ... | ... | ... |
| 989000 | 94.86 | 94.66 | 86.0 | 95.0 | 189.326 | 96.0 | 0.0 |
| 990000 | 94.66 | 94.65 | 85.0 | 95.0 | 187.015 | 94.0 | 0.0 |
| 991000 | 94.86 | 94.65 | 88.0 | 95.0 | 190.343 | 97.0 | 0.0 |
| 992000 | 94.91 | 94.65 | 91.0 | 95.0 | 189.35 | 96.0 | 0.0 |
| 993000 | 94.8 | 94.65 | 90.0 | 95.0 | 186.113 | 93.0 | 0.0 |
| 994000 | 94.88 | 94.65 | 90.0 | 95.0 | 190.361 | 97.0 | 0.0 |
| 995000 | 94.13 | 94.64 | 73.0 | 95.0 | 182.311 | 90.0 | 0.0 |
| 996000 | 94.55 | 94.63 | 84.0 | 95.0 | 184.822 | 92.0 | 0.0 |
| 997000 | 94.48 | 94.61 | 74.0 | 95.0 | 183.715 | 91.0 | 0.0 |
| 998000 | 94.88 | 94.66 | 90.0 | 95.0 | 189.312 | 96.0 | 0.0 |
| 999000 | 94.26 | 94.64 | 84.0 | 95.0 | 179.326 | 87.0 | 0.0 |
| 1000000 | 94.91 | 94.65 | 92.0 | 95.0 | 189.339 | 96.0 | 0.0 |
