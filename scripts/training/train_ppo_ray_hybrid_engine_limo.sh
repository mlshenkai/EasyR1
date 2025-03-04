set -x
export CUDA_VISIBLE_DEVICES=1,3,4,5,6,7
ray start --head --node-ip-address 127.0.0.1 --num-gpus 6

sleep 10s

yes | ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json='{"working_dir": "/tmp/code"}' \
   -- python3 -m openrlhf.cli.train_ppo_ray \
   --ref_num_nodes 1 \
   --ref_num_gpus_per_node 2 \
   --actor_num_nodes 1 \
   --actor_num_gpus_per_node 2 \
   --vllm_num_engines 1 \
   --vllm_tensor_parallel_size 2 \
   --init_kl_coef 1e-3 \
   --gamma 1.0 \
   --use_kl_loss \
   --kl_estimator k3 \
   --pretrain /code-online/code/EasyR1/resources/models/Qwen2.5-7B-Instruct1M \
   --save_path /tmp/code/examples/test_scripts/final/Qwen2.5-7B-rlhf \
   --ckpt_path /tmp/code/examples/test_scripts/ckpt/Qwen2.5-7B-rlhf \
   --save_hf_ckpt \
   --micro_train_batch_size 2 \
   --train_batch_size 32 \
   --micro_rollout_batch_size 2 \
   --rollout_batch_size 64 \
   --n_samples_per_prompt 1 \
   --max_epochs 1 \
   --num_episodes 1000 \
   --prompt_max_len 2048 \
   --max_samples 100000 \
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
   --temperature 0.7 \
   --advantage_estimator reinforce \
   --flash_attn \
   --save_steps 20 \
   --max_ckpt_num 3 \
   --vllm_enable_sleep \
   --adam_offload

#   --deepspeed_enable_sleep \

