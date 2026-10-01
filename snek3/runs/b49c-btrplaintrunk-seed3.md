# b49c-btrplaintrunk-seed3

step **879,000** · 879 evals · trailing **94.46** · peak **94.55** @841,000 · sef **35.3** · best30 **94.6** @638,000

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
| seed | 3 |
| target_update_period | 500 |
| target_update_tau | 1.0 |
| torch_threads | 1 |

![b49c-btrplaintrunk-seed3](b49c-btrplaintrunk-seed3.png)

## Evals

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 1000 | 3.53 | 3.53 | 0.0 | 16.0 | 2.895 | 0.0 | 0.8776 |
| 2000 | 15.62 | 9.57 | 3.0 | 30.0 | 10.509 | 0.0 | 0.85028 |
| 3000 | 13.59 | 10.91 | 1.0 | 25.0 | 8.71 | 0.0 | 0.82381 |
| ... | ... | ... | ... | ... | ... | ... | ... |
| 868000 | 94.51 | 94.45 | 83.0 | 95.0 | 180.598 | 88.0 | 0.0 |
| 869000 | 94.62 | 94.44 | 80.0 | 95.0 | 183.848 | 91.0 | 0.0 |
| 870000 | 94.71 | 94.45 | 88.0 | 95.0 | 183.934 | 91.0 | 0.0 |
| 871000 | 93.2 | 94.4 | 8.0 | 95.0 | 181.469 | 90.0 | 0.0 |
| 872000 | 94.72 | 94.42 | 79.0 | 95.0 | 190.199 | 97.0 | 0.0 |
| 873000 | 94.48 | 94.43 | 83.0 | 95.0 | 180.566 | 88.0 | 0.0 |
| 874000 | 94.9 | 94.45 | 92.0 | 95.0 | 189.335 | 96.0 | 0.0 |
| 875000 | 94.66 | 94.48 | 87.0 | 95.0 | 182.829 | 90.0 | 0.0 |
| 876000 | 94.74 | 94.51 | 87.0 | 95.0 | 187.098 | 94.0 | 0.0 |
| 877000 | 94.66 | 94.52 | 89.0 | 95.0 | 182.837 | 90.0 | 0.0 |
| 878000 | 94.15 | 94.49 | 45.0 | 95.0 | 184.411 | 92.0 | 0.0 |
| 879000 | 93.51 | 94.46 | 4.0 | 95.0 | 182.783 | 91.0 | 0.0 |
