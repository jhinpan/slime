import copy
import json
import random
import time
import uuid
from functools import partial
from multiprocessing import Process, Queue
from time import sleep
from typing import List, Optional

import requests
from generator.reward_utils.kernel_utils import get_kernel_code_reward, get_detailed_kernel_feedback
from openai import OpenAI
from tqdm import tqdm

TASK_TYPE = "kernelbench"

SAMPLING_PARAMS = {
    "top_p": 0.95,
    "temperature": 0.8,
}


def query_kernel_generation(client, messages, sampling_params, tools=None):
    """
    Query function specifically for kernel code generation.
    Handles multi-turn generation for complex CUDA kernels.
    """
    base_payload = {
        "messages": messages,
        **sampling_params,
        "model": "custom",
        "stream": False,
        "seed": random.randint(1, 10000000),
        "tools": tools,
    }

    text = None
    accumulated_tokens = 0
    max_attempts = 6

    for attempt in range(max_attempts):
        try:
            # Create a fresh payload for each attempt
            current_payload = copy.deepcopy(base_payload)

            if text is not None:
                # Update messages with current progress
                current_messages = copy.deepcopy(messages)
                current_messages.append({"role": "assistant", "content": text})
                current_payload["messages"] = current_messages

                # Adjust max_tokens based on accumulated tokens
                if "max_tokens" in sampling_params:
                    current_payload["max_tokens"] = max(0, sampling_params["max_tokens"] - accumulated_tokens)

                # Add continue flag for partial rollouts
                current_payload["extra_body"] = {"continue_final_message": True}
            
            if current_payload.get("max_tokens", 1) == 0:
                break
                
            response = client.chat.completions.create(**current_payload)

            if len(response.choices) > 0:
                if response.choices[0].finish_reason == "abort":
                    print(
                        f"Kernel generation query failed, reason: {response.choices[0].finish_reason}, "
                        f"currently generated: {response.usage.completion_tokens}"
                    )
                    accumulated_tokens += response.usage.completion_tokens

                    if text is None:
                        text = response.choices[0].message.content
                    else:
                        text += response.choices[0].message.content

                    sleep(10)
                    continue
                    
                if text is None:
                    text = response.choices[0].message.content
                elif response.choices[0].message.content is not None:
                    text += response.choices[0].message.content
                break
            else:
                print(f"Error in kernel generation query, status code: {response.status_code}")
                continue
                
        except Exception as e:
            print(f"Kernel generation query failed, error: {e}")
            if attempt < max_attempts - 1:
                sleep(5)
            continue

    # Update final messages
    if len(messages) > 0 and messages[-1]["role"] == "assistant":
        messages = messages[:-1]
    
    if text:
        messages.append({"role": "assistant", "content": text})
    else:
        # Fallback if generation completely failed
        messages.append({"role": "assistant", "content": "// Failed to generate kernel code"})

    return messages


def worker_process(task_queue, done_queue, rollout_func, reward_func, client, sampling_params):
    """Worker process for kernel code generation."""
    
    for line in iter(task_queue.get, "STOP"):
        if isinstance(line, str):
            item = json.loads(line)
        else:
            item = line

        try:
            # Generate kernel code
            messages = rollout_func(client, item["prompt"], sampling_params)

            item["uid"] = str(uuid.uuid4())
            item["messages"] = messages
            
            # Calculate reward
            reward = reward_func(item)
            
            # Get detailed feedback for extra_info
            feedback = get_detailed_kernel_feedback(item)
            
            item["rollout_index"] = 1
            item["reward"] = reward
            item["extra_info"] = {
                "cuda_syntax_valid": feedback["cuda_syntax_valid"],
                "python_syntax_valid": feedback["python_syntax_valid"],
                "has_required_functions": feedback["has_required_functions"],
                "cuda_complete": feedback["cuda_complete"],
                "optimization_techniques": feedback["optimization_techniques"],
                "error_count": len(feedback["errors"]),
            }
            item.update(sampling_params)
            item["timestamp"] = str(time.time())
            item["round_number"] = len([_ for _ in item["messages"] if _["role"] == "assistant"])

            output_item = {
                "uid": item.pop("uid"),
                "messages": messages,
                "reward": reward,
                "instance_id": item.pop("instance_id"),
                "extra_info": item,
            }

            done_queue.put(output_item)
            
        except Exception as e:
            print(f"Error processing kernel generation task: {e}")
            # Still put a failed item to maintain count
            output_item = {
                "uid": str(uuid.uuid4()),
                "messages": item.get("messages", item.get("prompt", [])),
                "reward": 0.0,
                "instance_id": item.get("instance_id", "unknown"),
                "extra_info": {"error": str(e)},
            }
            done_queue.put(output_item)

    done_queue.put("COMPLETE")


class KernelGenerator:
    def __init__(
        self,
        remote_engine_url,
        remote_buffer_url,
        num_repeat_per_sample=1,
        queue_size=1000000,
        num_process=10,
        task_type="kernelbench",
        max_tokens=8192,  # Larger for code generation
        num_repeats=10,
        skip_instance_ids: Optional[List[str]] = None,
    ):
        self.queue_size = queue_size
        self.num_process = num_process
        self.remote_engine_url = remote_engine_url
        self.remote_buffer_url = remote_buffer_url
        self.num_repeat_per_sample = num_repeat_per_sample
        self.task_type = task_type
        self.max_tokens = max_tokens
        self.num_repeats = num_repeats
        self.skip_instance_ids = list(skip_instance_ids) if skip_instance_ids is not None else None

        # Initialize OpenAI client
        self.client = OpenAI(
            base_url=f"{self.remote_engine_url}/v1",
            api_key="EMPTY",
        )

    def send_data_to_buffer(self, data):
        """Send generated kernel code data to buffer."""
        data_to_send = {
            "instance_id": data["instance_id"],
            "uid": data["uid"],
            "messages": data["messages"],
            "reward": data["reward"],
            "extra_info": data["extra_info"],
            "task_type": self.task_type,
        }
        
        # Debug: Check if assistant response is non-empty
        last_msg = data_to_send["messages"][-1]
        if last_msg.get("role") == "assistant" and not last_msg.get("content", "").strip():
            print(f"[WARNING] Empty assistant response for instance_id: {data_to_send['instance_id']}")
        
        response = requests.post(f"{self.remote_buffer_url}/buffer/write", json=data_to_send)
        if response.status_code != 200:
            print(f"Failed to send data to buffer: {response.text}")

    def run(self, input_file, rollout_func, reward_func):
        """Run kernel code generation rollout."""
        task_queue = Queue(self.queue_size)
        done_queue = Queue(self.queue_size)

        def read_data_into_queue():
            cnt = 0
            skipped_count = 0
            
            with open(input_file, "r") as f:
                for line in f:
                    for _ in range(self.num_repeat_per_sample):
                        item_repeat = copy.deepcopy(json.loads(line))

                        if "uid" not in item_repeat:
                            item_repeat["uid"] = str(uuid.uuid4())

                        # Check if instance_id should be skipped
                        if self.skip_instance_ids is not None and item_repeat["instance_id"] in self.skip_instance_ids:
                            print(f"Skipping instance_id: {item_repeat['instance_id']}")
                            self.skip_instance_ids.remove(item_repeat["instance_id"])
                            skipped_count += 1
                            continue

                        task_queue.put(item_repeat)
                        if cnt % 10 == 0:
                            print(f"[KernelGen] Queued {cnt * self.num_repeat_per_sample} samples from {cnt} prompts")
                    cnt += 1
                    
                # Add some delay to allow processing
                time.sleep(300)

            if skipped_count > 0:
                remaining_skip_count = len(self.skip_instance_ids) if self.skip_instance_ids is not None else 0
                print(
                    f"Kernel rollout summary: skipped {skipped_count} instance_ids, "
                    f"{remaining_skip_count} still in skip list"
                )

            for _ in range(self.num_process):
                task_queue.put("STOP")

        processes = []
        SAMPLING_PARAMS["max_tokens"] = self.max_tokens

        for _ in range(self.num_process):
            process = Process(
                target=partial(worker_process, client=self.client, sampling_params=SAMPLING_PARAMS),
                args=(task_queue, done_queue, rollout_func, reward_func),
            )
            process.start()
            processes.append(process)

        process = Process(target=read_data_into_queue)
        process.start()

        progress_bar = tqdm(desc="Generating kernel code")
        print("----- Starting Kernel Code Generation -----")

        num_finished = 0
        successful_generations = 0
        failed_generations = 0
        
        while num_finished < self.num_process:
            item = done_queue.get()
            if item == "COMPLETE":
                num_finished += 1
            else:
                assert "reward" in item, f"reward not in item: {item}"
                assert "instance_id" in item, f"instance_id not in item: {item}"
                
                if item["reward"] > 0:
                    successful_generations += 1
                else:
                    failed_generations += 1
                    
                self.send_data_to_buffer(item)
                progress_bar.update(1)
                if progress_bar.n % 20 == 0:
                    print(f"[KernelGen] Sent {progress_bar.n} samples to buffer, instance_id: {item.get('instance_id', 'unknown')}")

        progress_bar.close()
        print(f"Kernel generation completed. Success: {successful_generations}, Failed: {failed_generations}")

        return "finished"

    def entry(self, input_file, rollout_func, reward_func, num_epoch=1):
        """Entry point for kernel generation."""
        for epoch in range(num_epoch):
            print(f"Starting kernel generation epoch {epoch + 1}/{num_epoch}")
            status = self.run(input_file, rollout_func, reward_func)


def run_rollout(data: dict):
    """Main entry point for kernel code generation rollout."""
    
    print(f"Starting kernel code generation rollout with data: {data}")

    rollout_func = query_kernel_generation
    reward_func = get_kernel_code_reward

    print(f"Waiting for 10 seconds for buffer server to start")
    time.sleep(10)
    
    global SAMPLING_PARAMS
    for k, v in data["sampling_params"].items():
        SAMPLING_PARAMS[k] = v
        print(f"Set {k} to {v}", type(v))

    generator = KernelGenerator(
        data["remote_engine_url"],
        data["remote_buffer_url"],
        num_repeat_per_sample=int(data["num_repeat_per_sample"]),
        queue_size=1000000,
        max_tokens=int(data["sampling_params"]["max_tokens"]),
        num_process=int(data.get("num_process", 100)),
        task_type=data["task_type"],
        skip_instance_ids=data.get("skip_instance_ids", None),
    )

    generator.entry(data["input_file"], rollout_func, reward_func, int(data.get("num_epoch", 1)))


def normalize_group_data(group, epsilon=1e-8, algo="grpo"):
    """Normalize rewards for kernel code generation."""
    print(f"Using kernel-specific normalization for group {group[0]}")

    assert algo == "grpo", "Only 'grpo' is supported for now."

    instance_id = group[0]
    data = group[1]
    rewards = [item["reward"] for item in data]
    valid_rewards = [r for r in rewards if r >= 0]  # Filter out error cases

    if not valid_rewards or set(valid_rewards) == {0}:
        print(f"[Info] All rewards zero in kernel group {instance_id}, skipping normalization.")
        normalized_rewards = rewards
    else:
        mean_reward = sum(valid_rewards) / len(valid_rewards)
        std_reward = (sum((r - mean_reward) ** 2 for r in valid_rewards) / len(valid_rewards)) ** 0.5

        if std_reward < epsilon:
            print(f"[Info] Zero variance in kernel group {instance_id}, setting all to 0.")
            normalized_rewards = [0.0 if r >= 0 else r for r in rewards]
        else:
            normalized_rewards = [
                (r - mean_reward) / (std_reward + epsilon) if r >= 0 else r for r in rewards
            ]

    for i, item in enumerate(data):
        item["reward"] = normalized_rewards[i]
        item["raw_reward"] = rewards[i]

    return (instance_id, data)


def filter_item(item: dict, task_type: str = "kernelbench") -> bool:
    """
    Filter individual kernel code generation items.
    Returns True if the item should be kept, False otherwise.
    """
    # Basic validation
    required_fields = {"instance_id", "reward", "messages"}
    if not all(field in item for field in required_fields):
        return False

    # Check reward validity
    if not isinstance(item["reward"], (int, float)):
        return False

    # For kernel code, we want to keep all attempts including failures for learning
    # But we filter out corrupted data
    if item["reward"] < 0:  # Error case
        return False

    # Check messages format
    if not isinstance(item["messages"], list) or len(item["messages"]) == 0:
        return False

    # Ensure there's at least one assistant response
    has_assistant_response = any(msg.get("role") == "assistant" for msg in item["messages"])
    if not has_assistant_response:
        return False

    return True


def get_group_data_meta_info(temp_data: dict) -> dict:
    """Get meta information for kernel code generation groups."""
    if not temp_data:
        return {
            "total_samples": 0,
            "num_groups": 0,
            "avg_group_size": 0,
            "avg_reward": 0,
            "successful_generations": 0,
            "syntax_valid_count": 0,
            "cuda_complete_count": 0,
        }

    meta_info = {
        "total_samples": 0,
        "num_groups": len(temp_data),
        "successful_generations": 0,
        "syntax_valid_count": 0,
        "cuda_complete_count": 0,
    }

    all_rewards = []
    
    for instance_id, samples in temp_data.items():
        group_size = len(samples)
        meta_info["total_samples"] += group_size
        
        for s in samples:
            reward = s.get("reward", 0)
            all_rewards.append(reward)
            
            if reward > 0.5:  # Consider as successful
                meta_info["successful_generations"] += 1
            
            # Check extra info
            extra_info = s.get("extra_info", {})
            if extra_info.get("python_syntax_valid", False) and extra_info.get("cuda_syntax_valid", False):
                meta_info["syntax_valid_count"] += 1
            if extra_info.get("cuda_complete", False):
                meta_info["cuda_complete_count"] += 1

    meta_info["avg_group_size"] = meta_info["total_samples"] / meta_info["num_groups"] if meta_info["num_groups"] > 0 else 0

    if all_rewards:
        meta_info["avg_reward"] = sum(all_rewards) / len(all_rewards)
        meta_info["success_rate"] = meta_info["successful_generations"] / len(all_rewards)
        meta_info["syntax_valid_rate"] = meta_info["syntax_valid_count"] / len(all_rewards)
        meta_info["cuda_complete_rate"] = meta_info["cuda_complete_count"] / len(all_rewards)
    else:
        meta_info["avg_reward"] = 0
        meta_info["success_rate"] = 0
        meta_info["syntax_valid_rate"] = 0
        meta_info["cuda_complete_rate"] = 0

    return meta_info