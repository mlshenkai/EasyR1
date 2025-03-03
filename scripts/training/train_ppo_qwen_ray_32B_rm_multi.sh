set -x

ray start --head --node-ip-address 127.0.0.1 --num-gpus 8

sleep 10s

yes | ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json='{"working_dir": "/tmp/code"}' \
   -- python3 -m openrlhf.cli.train_ppo_ray \
   --ref_num_nodes 1 \
   --ref_num_gpus_per_node 2 \
   --reward_num_nodes 1 \
   --reward_num_gpus_per_node 2 \
   --actor_num_nodes 1 \
   --actor_num_gpus_per_node 2 \
   --vllm_num_engines 2 \
   --vllm_tensor_parallel_size 2 \
   --vllm_gpu_memory_utilization 0.9 \
   --init_kl_coef 1e-3 \
   --gamma 1.0 \
   --use_kl_loss \
   --kl_estimator k2 \
   --pretrain /llm/qwen/Qwen2.5-7B-Instruct \
   --save_path /tmp/code/examples/test_scripts/final/Qwen2.5-7B-rlhf \
   --ckpt_path /tmp/code/examples/test_scripts/ckpt/Qwen2.5-7B-rlhf \
   --save_hf_ckpt \
   --micro_train_batch_size 2 \
   --train_batch_size 64 \
   --micro_rollout_batch_size 4 \
   --rollout_batch_size 512 \
   --n_samples_per_prompt 1 \
   --max_epochs 1 \
   --prompt_max_len 2048 \
   --max_samples 100000 \
   --prompt_max_len 2048 \
   --generate_max_len 4096 \
   --zero_stage 3 \
   --bf16 \
   --actor_learning_rate 2e-7 \
   --critic_learning_rate 2e-6 \
   --prompt_data /code-online/code/EasyR1/data/aime/aime_formatted_qwen.jsonl \
   --input_key input \
   --normalize_reward \
   --gradient_checkpointing \
   --packing_samples \
   --vllm_sync_backend nccl \
   --enforce_eager \
   --use_wandb "b4d97196d9460a5f190eef95e38a06518badc099" \
   --wandb_project "limo_grpo" \
   --remote_rm_url http://127.0.0.1:8000/get_reward \
   --advantage_estimator reinforce \
