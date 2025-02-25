# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Preprocess the GSM8k dataset to parquet format
"""
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

import os
import datasets

from verl.utils.hdfs_io import copy, makedirs
import argparse
from transformers import AutoTokenizer
from verl.utils.reward_score.math import remove_boxed, last_boxed_only_string


def make_prompt(question: str, tokenizer) -> str:
    messages = [
        {"role": "system", "content": "You are a helpful assistant. When answering, think step by step. Enclose the reasoning process inside <think></think> tags, and the answer inside <answer></answer> tags. The final result should be enclosed in \\boxed{}. i.e., <think> reasoning process here </think><answer> answer here </answer>."},
        {"role": "user", "content": question},
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    return text

def extract_solution(solution_str):
    return remove_boxed(last_boxed_only_string(solution_str))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_dir', default='/code-online/code/EasyR1/data/numina_math')
    parser.add_argument('--hdfs_dir', default=None)

    args = parser.parse_args()

    # 'lighteval/MATH' is no longer available on huggingface.
    # Use mirror repo: DigitalLearningGmbH/MATH-lighteval
    data_source = 'AI-MO/NuminaMath-TIR'
    print(f"Loading the {data_source} dataset from huggingface...", flush=True)
    dataset = datasets.load_dataset(data_source, trust_remote_code=True)
    model_name = "/code-online/code/EasyR1/resources/models/Qwen2.5-7B-Instruct1M"
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    train_dataset = dataset['train']
    test_dataset = dataset['test']


    # add a row to each data item that represents a unique id
    def make_map_fn(split):

        def process_fn(example, idx):
            question = example.pop('problem')

            # question = make_prompt(question, tokenizer)

            answer = example.pop('solution')
            solution = extract_solution(answer)
            data = {
                "data_source": data_source,
                "prompt": [
                    {
                        "role": "system",
                        "content": "You are a helpful assistant. When answering, think step by step. Enclose the reasoning process inside <think></think> tags, and the answer inside <answer></answer> tags. The final result should be part of the <answer> tag, and should be expressed as part of the complete answer, including the boxed final result in the form of \\boxed{final answer}. For example:\n <think> reasoning process here </think>\n<answer> reasoning conclusion and final result: \\boxed{final answer} </answer>"
                    },
                    {
                        "role": "user",
                        "content": question
                    }],
                "ability": "math",
                "reward_model": {
                    "style": "rule",
                    "ground_truth": solution
                },
                "extra_info": {
                    'split': split,
                    'index': idx
                }
            }
            return data

        return process_fn

    train_dataset = train_dataset.map(function=make_map_fn('train'), with_indices=True)
    test_dataset = test_dataset.map(function=make_map_fn('test'), with_indices=True)

    local_dir = args.local_dir
    hdfs_dir = args.hdfs_dir

    train_dataset.to_parquet(os.path.join(local_dir, 'train.parquet'))
    test_dataset.to_parquet(os.path.join(local_dir, 'test.parquet'))

    if hdfs_dir is not None:
        makedirs(hdfs_dir)

        copy(src=local_dir, dst=hdfs_dir)
