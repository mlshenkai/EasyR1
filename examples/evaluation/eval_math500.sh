export HF_HOME=/code-online/cache/huggingface
export HF_ENDPOINT=https://hf-mirror.com
export CUDA_VISIBLE_DEVICES=0
NUM_GPUS=1 # Set to 8 for 32B and 70B models
MODEL=/code-online/code/EasyR1/resources/models/DeepSeek-R1-Distill-Qwen-1.5B
MODEL_ARGS="pretrained=$MODEL,dtype=bfloat16,max_model_length=32768,gpu_memory_utilization=0.8,data_parallel_size=$NUM_GPUS,generation_parameters={max_new_tokens:32768,temperature:0.6,top_p:0.95}"
OUTPUT_DIR=data/evals/DeepSeek-R1-Distill-Qwen-1.5B

lighteval vllm $MODEL_ARGS "custom|math_500|0|0" \
    --custom-tasks evaluation/lighteval_evaluate.py \
    --use-chat-template \
    --output-dir $OUTPUT_DIR