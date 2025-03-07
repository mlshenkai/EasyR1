# -*- coding: utf-8 -*-
# @Author: watcher
# @Created Time: 2025/3/7 10:40
# @File: orm_server
# @Email: mlshenkai@163.com
import asyncio
import json
import os
import re
import signal
import time
import argparse
import datetime
from http.client import HTTPException
from pathlib import Path
from typing import List
import concurrent.futures
import uvicorn
from datasets import load_dataset, Dataset, DatasetDict
from transformers import AutoTokenizer
import logging
from fastapi import FastAPI, Request, HTTPException

from evaluation.grader import math_equal
from rewards.reward_funcs.deepseek_reward import grade_cot

# 设定日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Globals (you may need them to be accessible by each process)
args = None
ground_truth_dict = {}
tokenizer = None


class RequestLogger:
    def __init__(self, log_dir="logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)

        # Create a new log file for each run with timestamp
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = self.log_dir / f"requests_{timestamp}.jsonl"

        # Ensure the log file exists
        self.log_file.touch()

        logger.info(f"Logging requests to: {self.log_file}")

    def log_request(
        self, request_data: dict, reward: float | List[float], processing_time: float
    ):
        """Log a request with its reward and processing time"""
        log_entry = {
            "timestamp": datetime.datetime.now().isoformat(),
            "request": request_data,
            "reward": reward,
            "processing_time": processing_time,
        }

        with open(self.log_file, "a") as f:
            f.write(json.dumps(log_entry) + "\n")


def parse_query(sequence):
    # Parse the query
    global args
    if "llama" in args.model_name.lower():
        query = (
            sequence.split("Problem: ")[1]
            .split("<|eot_id|><|start_header_id|>assistant<|end_header_id|>")[0]
            .strip()
        )
        response = sequence.split("<|start_header_id|>assistant<|end_header_id|>\n\n")[
            1
        ].strip()
    elif "qwen" in args.model_name.lower():
        # <|im_start|>user\nProblem:{input}<|im_end|>\n<|im_start|>assistant\n
        query = (
            sequence.split("<|im_start|>user\nProblem:")[1]
            .split("<|im_end|>")[0]
            .strip()
        )
        response = sequence.split("<|im_start|>assistant\n")[1].strip()
    elif "cot" in args.model_name.lower():
        query = sequence.split("Question: ")[1].split("\nAnswer: ")[0].strip()
        response = sequence.split("Answer: ")[1].strip()
    else:
        raise ValueError("Model name not recognized")
    return query, response


def extract_answer_and_reasoning(text):
    # Define the regex pattern to match the outermost \boxed{}
    pattern = r"\\boxed{([^{}]*(?:{[^{}]*}[^{}]*)*)}"

    # Search for the pattern in the text
    match = re.search(pattern, text)

    # If a match is found, return the matched group, otherwise return an empty string
    if match:
        return match.group(1), text[: match.start()]
    else:
        return None, None


def _compute_single_reward(seq: str) -> float:
    """
    A helper function that computes the reward for a single sequence.
    This function can be safely called in parallel worker processes.
    """
    global args, ground_truth_dict, tokenizer

    try:
        tic = time.time()
        # Parse query & response
        query, response = parse_query(seq)
        ground_truth = ground_truth_dict[re.sub(r"[^a-zA-Z0-9]", "", query)]
        print("\n========response========\n")
        print(response)
        print("\n========================\n")
        answer, reasoning = extract_answer_and_reasoning(response)
        print("\n========ground_true:answer========\n")
        print(ground_truth, answer)
        print("\n========================\n")
        if answer is None:
            # means boxed not in response
            reward = -0.5
        else:
            # Base reward: math_equal
            reward = math_equal(ground_truth, answer, timeout=True) + 0.0

        # Apply length penalty if specified
        if args.length_penalty != 0.0:
            if reasoning is None:
                reasoning_length = len(response)
            else:
                reasoning_length = len(reasoning)
            # Subtract a fraction of the length penalty
            reward = reward - args.length_penalty / reasoning_length

        duration = time.time() - tic
        print(time.ctime(), f"GT Reward = {reward:.3f}; GT Duration: {duration:.2f}s")

        # Optionally use GPT-based reward
        if args.use_gpt:
            tic = time.time()
            aux_reward = grade_cot(
                query,
                response,
                grader_model_name="deepseek-r1",
                api_type="deepseek",
                endpoint_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
                temperature=0.6,
            )
            duration = time.time() - tic
            print(
                time.ctime(),
                f"GT Reward = {reward:.3f}, GPT Reward = {aux_reward:.3f}; GPT Duration: {duration:.2f}s",
            )

            # If GPT fails, use 0.5 as fallback
            if aux_reward == -1:
                aux_reward = 0.5
            # Combine GPT reward with math reward
            reward = reward * 0.8 + aux_reward * 0.2

    except Exception as e:
        # If any error occurs, default to 0.0
        reward = 0.0

    return reward


def calculate_reward(data: dict) -> float | list[float]:
    """
    Calculate rewards in parallel for each sequence in data["query"].
    """
    global args

    sequence = data.get("query", "")
    if isinstance(sequence, str):
        sequence = [sequence]

    # Use a process pool to compute each reward in parallel
    with concurrent.futures.ProcessPoolExecutor() as executor:
        rewards = list(executor.map(_compute_single_reward, sequence))
    return rewards


# FastAPI 实例
app = FastAPI()

# 监听终止信号
stop_event = asyncio.Event()


def shutdown():
    logger.info("Received shutdown signal. Exiting...")
    stop_event.set()


signal.signal(signal.SIGTERM, lambda sig, frame: shutdown())
signal.signal(signal.SIGINT, lambda sig, frame: shutdown())


@app.on_event("startup")
async def startup_event():
    try:
        app.state.request_logger = RequestLogger(log_dir=args.log_dir)
        logger.info("Request logger initialized")
    except Exception as e:
        logger.critical(f"Failed to initialize request logger: {e}", exc_info=True)
        raise RuntimeError("Server startup failed")


@app.on_event("shutdown")
async def shutdown_event():
    logger.info("Server is shutting down...")


@app.post("/get_reward")
async def get_reward(request: Request):
    start_time = time.time()
    try:
        json_data = await request.json()
        if not isinstance(json_data, dict):
            raise ValueError("Invalid request format")

        # **确保计算任务不会阻塞**
        loop = asyncio.get_running_loop()
        reward = await loop.run_in_executor(None, calculate_reward, json_data)

        # 记录日志
        processing_time = time.time() - start_time
        if hasattr(app.state, "request_logger"):
            loop.run_in_executor(
                None,
                app.state.request_logger.log_request,
                json_data,
                reward,
                processing_time,
            )
        else:
            logger.warning("Request logger is not initialized")

        return {"rewards": reward}

    except ValueError as ve:
        logger.warning(f"Invalid request data: {ve}")
        raise HTTPException(status_code=400, detail=str(ve))

    except Exception as e:
        logger.error(f"Unhandled error: {e}", exc_info=True)
        return {"rewards": 0.0, "error": str(e)}


# **优化数据加载，避免 OOM**
def load_ground_truth(dataset_path):
    ground_truth_dict = {}
    try:
        dataset = load_dataset(dataset_path)
        if isinstance(dataset, (DatasetDict, Dataset)):
            for split in dataset if isinstance(dataset, DatasetDict) else [dataset]:
                for example in dataset[split]:
                    query = re.sub(r"[^a-zA-Z0-9]", "", example["problem"])
                    ground_truth_dict[query] = example["answer"]
        else:
            raise ValueError("Dataset format not recognized")
    except Exception as e:
        logger.critical(f"Failed to load dataset: {e}", exc_info=True)
        raise RuntimeError("Dataset loading failed")

    return ground_truth_dict


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run the Code Contests Reward Model server"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="/code-online/code/EasyR1/evaluation/data/aime_full_except_24",
    )
    parser.add_argument(
        "--model_name", type=str, default="/llm/qwen/Qwen2.5-32B-Instruct"
    )
    parser.add_argument("--log_dir", type=str, default="logs")
    parser.add_argument("--length_penalty", type=float, default=20.0)
    parser.add_argument("--use_gpt", action="store_true", help="Use GPT or not")

    args = parser.parse_args()
    os.makedirs(args.log_dir, exist_ok=True)

    # **优化 tokenizer 加载，避免内存泄漏**
    tokenizer = None
    try:
        if args.length_penalty != 0.0:
            tokenizer = AutoTokenizer.from_pretrained(
                args.model_name, low_cpu_mem_usage=True
            )
    except Exception as e:
        logger.error(f"Failed to load tokenizer: {e}", exc_info=True)

    # **优化 ground truth 数据加载**
    ground_truth_dict = {}
    try:
        ground_truth_dict = load_ground_truth(args.dataset)
    except RuntimeError:
        logger.critical("Server cannot start due to dataset loading failure")
        exit(1)

    # **使用 `uvicorn` 运行，监听终止信号**
    config = uvicorn.Config(app, host="0.0.0.0", port=8000, log_level="info")
    server = uvicorn.Server(config)

    loop = asyncio.get_event_loop()
    loop.create_task(server.serve())

    try:
        loop.run_until_complete(stop_event.wait())  # **等待 SIGTERM 终止信号**
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received, shutting down...")
    finally:
        loop.run_until_complete(server.shutdown())
        logger.info("Server stopped.")
