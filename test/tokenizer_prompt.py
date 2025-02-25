# -*- coding: utf-8 -*-
# @Author: watcher
# @Created Time: 2025/2/13 11:57
# @File: tokenizer_prompt
# @Email: mlshenkai@163.com
from transformers import AutoTokenizer, AutoModelForCausalLM
model_name = "/code-online/code/EasyR1/resources/models/Qwen2.5-7B-Instruct1M"
tokenizer = AutoTokenizer.from_pretrained(model_name)
prompt = "你好"
messages = [
    {"role": "system", "content": "You are a helpful assistant. When answering, think step by step. Enclose the reasoning process inside <think></think> tags, and the answer inside <answer></answer> tags. The final result should be enclosed in \\boxed{}. i.e., <think> reasoning process here </think><answer> answer here </answer>."},
    {"role": "user", "content": prompt},
]
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)
print(text)