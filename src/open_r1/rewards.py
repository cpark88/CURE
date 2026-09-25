# coding=utf-8
# Copyright 2025 The HuggingFace Team. All rights reserved.
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

"""Reward functions for GRPO training."""

import asyncio
import json
import math
import re
from functools import partial, update_wrapper
from typing import Callable, Dict, Optional

import re
from typing import Optional


###
from sasrec.util import neg_sample, neg_sample_set, get_sample_scores, neg_sample_unigram
from sasrec.customized_model import SASRecModel
from typing import Union, Dict, Optional, Sequence
import numpy as np
import os

import copy
import logging
from dataclasses import dataclass

import torch
import transformers
import sasrec.util
from torch.utils.data import Dataset

from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
import wandb
import numpy as np
import dataclasses
import logging
import math
import os
import io
import sys
import time
import json
from typing import Optional, Sequence, Union
import tqdm
import copy

from sasrec.util import dict_str_key_to_int, sequential_loss, clm_loss, sequential_loss_item, kl_divergence_loss, AdaptiveLossWeighting, normalize_loss, init_weights
from sasrec.sequential_reco import *
from sasrec.customized_model import *
from sasrec.outputs import ModelArguments, DataArguments, TrainingArguments, MyCallback
from sasrec.util import smart_tokenizer_and_embedding_resize_v3, dict_str_key_to_int, CustomTrainer, CustomCallback
from safetensors.torch import load_file
import json
from typing import Union, Dict, Optional, Sequence
import tqdm
import numpy as np
import os

import copy
import logging

import torch
import transformers
from torch.utils.data import Dataset
from transformers import Trainer


from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
import wandb
import numpy as np
from safetensors import safe_open

# For embedding similarity reward
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    print("Warning: sentence-transformers not available. embedding_similarity_reward will not work.")


def conditional_sc_reward(completions: list[list[dict[str, str]]], output: list[str], output_id: list[str], self_certainty: list[Optional[float]], product_type: list[str], cross_product_type: list[str], prompts: Optional[list[list[dict[str, str]]]] = None, gamma: float = 0.5, epsilon: float = 1e-8, **kwargs) -> list[Optional[float]]:
    """
    conditional_sc_reward
    
    For item_r == 1 and a_domain != b_domain (item matching and cross-domain case),
    uses embedding_similarity_reward with calibration:
    - reward = gamma * similarity + (1-gamma) * item_r
    - calibrated_reward = clip(a * reward + b, 0, 1)
    where a = sigma_in / (sigma_cr + epsilon), b = mu_in - a * mu_cr
    """
    item_match_rewards = item_id_match_reward(completions, output) # 1 or 0
    # self_certainty = self_certainty # B
    assert len(item_match_rewards) == len(self_certainty), "different lengths between self_certainty and item!"
    
    print("#"*10)
    print(self_certainty)
    print("#"*10)
    
    
    print("$"*10)
    print(item_match_rewards)
    print("$"*10)
    
    # Compute embedding similarities - always compute if prompts are available
    embedding_similarities = None
    if prompts is not None and SENTENCE_TRANSFORMERS_AVAILABLE:
        try:
            embedding_similarities = embedding_similarity_reward(
                completions=completions,
                prompts=prompts,
                **kwargs
            )
            # Validate that we got a valid list
            if embedding_similarities is None or not isinstance(embedding_similarities, list):
                raise ValueError(f"embedding_similarity_reward returned invalid value: {type(embedding_similarities)}")
            if len(embedding_similarities) != len(item_match_rewards):
                raise ValueError(f"embedding_similarities length ({len(embedding_similarities)}) doesn't match item_match_rewards length ({len(item_match_rewards)})")
        except Exception as e:
            print(f"Error: Failed to compute embedding similarity in conditional_sc_reward: {e}")
            import traceback
            traceback.print_exc()
            raise  # Re-raise the exception since we require embedding similarity
    elif prompts is None:
        raise ValueError("prompts is required for conditional_sc_reward when using embedding similarity")
    elif not SENTENCE_TRANSFORMERS_AVAILABLE:
        raise ImportError("sentence-transformers is required for conditional_sc_reward")
    
    # First pass: compute raw rewards
    raw_rewards = []
    in_domain_rewards = []  # item_r == 1 and a_domain == b_domain
    cross_domain_rewards = []  # item_r == 1 and a_domain != b_domain (raw, before calibration)
    cross_domain_indices = []  # indices of cross-domain cases
    
    for idx, (item_r, sc, a_domain, b_domain) in enumerate(zip(item_match_rewards, self_certainty, product_type, cross_product_type)):
        if item_r == 1 and a_domain == b_domain: # item matching and in-domain case
            reward = item_r
            in_domain_rewards.append(reward)
            raw_rewards.append(reward)
        elif item_r == 1 and a_domain != b_domain: # item matching and cross-domain case
            # Compute reward = gamma * similarity + (1-gamma) * item_r
            if embedding_similarities is None:
                raise ValueError("embedding_similarities is required for cross-domain matching cases")
            if not isinstance(embedding_similarities, list) or idx >= len(embedding_similarities):
                raise ValueError(f"Invalid embedding_similarities access at index {idx}")
            
            similarity_val = embedding_similarities[idx]
            if similarity_val is None:
                raise ValueError(f"embedding_similarities[{idx}] is None")
            
            try:
                similarity_float = float(similarity_val)
                # reward = gamma * similarity_float + (1 - gamma) * item_r #ours
                reward = item_r
                cross_domain_rewards.append(reward)
                cross_domain_indices.append(idx)
                raw_rewards.append(reward)
            except (TypeError, ValueError) as e:
                raise ValueError(f"Failed to convert embedding_similarities[{idx}] to float: {e}")
        elif item_r == 0 and a_domain == b_domain: # item non-matching and in-domain case
            reward = 0.0
            raw_rewards.append(reward)
        elif item_r == 0 and a_domain != b_domain: # item non-matching and cross-domain case
            reward = 0.0
            raw_rewards.append(reward)
        else:
            reward = 0.0
            raw_rewards.append(reward)
    
    # Compute statistics for calibration
    if len(in_domain_rewards) > 0 and len(cross_domain_rewards) > 0:
        mu_in = np.mean(in_domain_rewards)
        sigma_in = np.std(in_domain_rewards, ddof=0)  # Population std
        mu_cr = np.mean(cross_domain_rewards)
        sigma_cr = np.std(cross_domain_rewards, ddof=0)  # Population std
        
        # Compute calibration parameters
        # a = sigma_in / (sigma_cr + epsilon)
        # b = mu_in - a * mu_cr
        a = sigma_in / (sigma_cr + epsilon)
        b = mu_in - a * mu_cr
        
        # Second pass: apply calibration to cross-domain rewards
        final_rewards = raw_rewards.copy()
        for idx in cross_domain_indices:
            raw_reward = raw_rewards[idx]
            calibrated_reward = a * raw_reward + b
            final_rewards[idx] = np.clip(calibrated_reward, 0.0, 1.0)
    else:
        # If no in-domain or cross-domain cases, use raw rewards
        final_rewards = raw_rewards

    return final_rewards


# irm_final_rewards    
def collaborative_guided_reward(completions: list[list[dict[str, str]]], output: list[str], output_id: list[str], lambda_: float = 3.0, cf_threshold: float = 0.3, **kwargs) -> list[Optional[float]]:
    """
    collaborative_guided_reward
    
    - item_rewards: 0 or 1
    - cf_rewards: float (0~1)
    - lambda_: weight for CF reward if item is matched
    - delta: penalty if CF is high but item is wrong
    - cf_threshold: threshold to define 'high' CF score
    """
    cf_rewards = collaborative_filtering_reward(completions, output_id)
    item_match_rewards = item_id_match_reward(completions, output)
    assert len(item_match_rewards) == len(cf_rewards), "different lengths between cf and item!"

    final_rewards = []
    for item_r, cf_r in zip(item_match_rewards, cf_rewards):
        if item_r == 1 and cf_r >= cf_threshold:
            reward = 1.0 + lambda_ * cf_r
        elif item_r == 1 and cf_r < cf_threshold:
            reward = 1.0
        elif item_r == 0 and cf_r >= cf_threshold:
            reward = cf_r#-delta
        else:  # item_r == 0 and cf_r < threshold
            reward = 0.0
        final_rewards.append(reward)

    return final_rewards    
    
    
    

def collaborative_filtering_reward(completions: list[list[dict[str, str]]], output_id: list[str], **kwargs) -> list[Optional[float]]: #solution
    """Reward function that checks if the text inside <item>...</item> exactly matches the solution."""
    contents = [completion[0]["content"] for completion in completions] # vLLM output format 
    rewards = []
    for content, id_ in zip(contents, output_id):#solution
        input_ids = torch.tensor([int(i) for i in id_.split(",")]).unsqueeze(0)
        input_ids_item = customized_pad_sequence(input_ids, batch_first=True, padding_value=2, pos='left') 
        match = re.search(r"<item_nm>(.*?)</item_nm>", content, re.DOTALL) #amazon
        if match:
            item_text = match.group(1).strip()
            try:
                llm_infer_answer_id = torch.tensor(int(vocab_dict_inverse[item_text])).unsqueeze(0)
                # SASRec logits
                with torch.no_grad():
                    print("input_id_item", input_ids_item)
                    print("answer_id", llm_infer_answer_id)
                    reward = sasrec_model.extract_logits(answer_id = llm_infer_answer_id.cuda(), test_neg=None, input_ids_item=input_ids_item.cuda() ).cpu().detach().numpy().copy()[0][0]
                    print("reward", reward)
            except:
                reward = 0.0
            
        else:
            reward = 0.0
        rewards.append(reward)
    return rewards   



def item_match_reward(completions: list[list[dict[str, str]]], output: list[str], **kwargs) -> list[Optional[float]]: #solution
    """Reward function that checks if the text inside <item>...</item> exactly matches the solution."""
    contents = [completion[0]["content"] for completion in completions] # vLLM output format 
    rewards = []
    for content, sol in zip(contents, output):#solution
        match = re.search(r"<item_nm>(.*?)</item_nm>", content, re.DOTALL) #amazon
        sol_match = re.search(r"<item_nm>(.*?)</item_nm>", sol, re.DOTALL) #amazon
        if match:
            item_text = match.group(1).strip()
            sol_text = sol_match.group(1).strip()
            # matching
            reward = 1.0 if item_text == sol_text else 0.0
        else:
            reward = 0.0
        rewards.append(reward)
    return rewards

def item_id_match_reward(completions: list[list[dict[str, str]]], output: list[str], **kwargs) -> list[Optional[float]]: #solution
    """Reward function that checks if the text inside <item>...</item> exactly matches the solution."""
    contents = [completion[0]["content"] for completion in completions] # vLLM output format 
    rewards = []
    for content, sol in zip(contents, output):#solution
        match = re.search(r"<item_id>(.*?)</item_id>", content, re.DOTALL) #amazon
        sol_match = re.search(r"<item_id>(.*?)</item_id>", sol, re.DOTALL) #amazon
        if match:
            item_text = match.group(1).strip()
            sol_text = sol_match.group(1).strip()
            reward = 1.0 if item_text == sol_text else 0.0
        else:
            reward = 0.0
        rewards.append(reward)

    return rewards   
    
def only_expected_tags_reward(completions: list[list[dict[str, str]]], **kwargs) -> list[Optional[float]]:
    """
    Reward is 1.0 only if the completion contains ONLY the allowed tags:
    <think>, <item>, <sales> (with matching closing tags),
    and NO other custom tags like <xxx>...</xxx>.
    """
    allowed_tags = {"think", "item_nm","item_id", "sales"}
    contents = [completion[0]["content"] for completion in completions]
    rewards = []

    for content in contents:
        tags = re.findall(r"</?([a-zA-Z0-9_]+)>", content)
        unique_tags = set(tags)
        if unique_tags.issubset(allowed_tags):
            reward = 1.0
        else:
            reward = 0.0

        rewards.append(reward)

    return rewards    
    
def tag_presence_reward(completions: list[list[dict[str, str]]], **kwargs) -> list[Optional[float]]:
    contents = [completion[0]["content"] for completion in completions]
    rewards = []
    required_tags = ["item_id", "item_nm", "think"]

    for content in contents:
        all_present = True
        for tag in required_tags:
            pattern = fr"<{tag}>.*?</{tag}>"
            if not re.search(pattern, content, re.DOTALL):
                all_present = False
                break
        reward = 1.0 if all_present else 0.0
        rewards.append(reward)

    return rewards    


def embedding_similarity_reward(
    completions: list[list[dict[str, str]]], 
    prompts: list[list[dict[str, str]]],
    model_name: str = "sentence-transformers/all-mpnet-base-v2",
    **kwargs
) -> list[Optional[float]]:
    """
    Reward function that computes embedding similarity between:
    1. User history from the prompt
    2. Recommended item + reasoning from the completion
    
    Args:
        completions: List of completions, each containing a dict with "content"
        prompts: List of prompts (conversational format)
        model_name: Sentence transformer model name
        **kwargs: Additional arguments (output, output_id, etc.)
    
    Returns:
        List of similarity scores (0-1 range, typically -1 to 1 for cosine similarity)
    """
    if not SENTENCE_TRANSFORMERS_AVAILABLE:
        return [0.0] * len(completions)
    
    # Initialize model (cache it to avoid reloading)
    if not hasattr(embedding_similarity_reward, '_model'):
        embedding_similarity_reward._model = SentenceTransformer(model_name, token=False)
        embedding_similarity_reward._model.eval()
    
    model = embedding_similarity_reward._model
    device = next(model.parameters()).device
    
    contents = [completion[0]["content"] for completion in completions]
    rewards = []
    
    # Extract user history from prompts and item+reasoning from completions
    user_histories = []
    item_reasonings = []
    
    for prompt, content in zip(prompts, contents):
        # Extract user history from prompt
        # Prompt format: [{"role": "system", "content": ...}, {"role": "user", "content": ...}]
        user_history_text = ""
        for msg in prompt:
            if msg.get("role") == "user":
                user_content = msg.get("content", "")
                # Extract [Purchase History] section if it exists
                history_match = re.search(r'\[Purchase History\](.*?)(?:\[Candidate List\]|$)', user_content, re.DOTALL)
                if history_match:
                    user_history_text = history_match.group(1).strip()
                else:
                    # If no [Purchase History] tag, use the full user content
                    user_history_text = user_content
                break
        
        # Extract item and reasoning from completion
        item_text = ""
        reasoning_text = ""
        
        # Extract reasoning
        reasoning_match = re.search(r'<think>(.*?)</think>', content, re.DOTALL)
        if reasoning_match:
            reasoning_text = reasoning_match.group(1).strip()
        
        # Extract item name
        item_match = re.search(r'<item_nm>(.*?)</item_nm>', content, re.DOTALL)
        if item_match:
            item_text = item_match.group(1).strip()
        
        # Combine item and reasoning
        item_reasoning_text = f"{item_text}. {reasoning_text}".strip()
        
        user_histories.append(user_history_text)
        item_reasonings.append(item_reasoning_text)
    print("User History Text:", user_histories[0])
    print("Item-Reasoning Text:", item_reasonings[0])
    
    # Compute embeddings
    try:
        with torch.no_grad():
            user_embeddings = model.encode(user_histories, convert_to_tensor=True, device=device)
            item_embeddings = model.encode(item_reasonings, convert_to_tensor=True, device=device)
        
        # Compute cosine similarity
        # Normalize embeddings
        user_embeddings = torch.nn.functional.normalize(user_embeddings, p=2, dim=1)
        item_embeddings = torch.nn.functional.normalize(item_embeddings, p=2, dim=1)
        
        # Cosine similarity
        similarities = (user_embeddings * item_embeddings).sum(dim=1).cpu().numpy()
        
        # Convert to list (similarity is typically -1 to 1, but we can normalize to 0-1)
        # Option 1: Use as-is (range -1 to 1)
        # Option 2: Normalize to 0-1: (similarity + 1) / 2
        # rewards = [(sim + 1) / 2 for sim in similarities]  # Normalize to 0-1
        # rewards = [torch.sigmoid(sim) for sim in similarities]
        
        tau = 5.0
        # rewards = [1 / (1 + np.exp(-tau * sim)) for sim in similarities] 
        # rewards = 1 / (1 + np.exp(-tau * similarities))
        rewards = (1 / (1 + np.exp(-tau * similarities))).tolist() 
        
    except Exception as e:
        print(f"Error computing embedding similarity: {e}")
        rewards = [0.0] * len(completions)
    
    return rewards


def accuracy_reward(completions: list[list[dict[str, str]]], solution: list[str], **kwargs) -> list[Optional[float]]:
    """Reward function that checks if the completion is the same as the ground truth."""
    contents = [completion[0]["content"] for completion in completions]
    rewards = []
    for content, sol in zip(contents, solution):
        gold_parsed = parse(
            sol,
            extraction_mode="first_match",
        )
        if len(gold_parsed) != 0:
            # We require the answer to be provided in correct latex (no malformed operators)
            answer_parsed = parse(
                content,
                extraction_config=[
                    LatexExtractionConfig(
                        normalization_config=NormalizationConfig(
                            nits=False,
                            malformed_operators=False,
                            basic_latex=True,
                            equations=True,
                            boxed="all",
                            units=True,
                        ),
                        # Ensures that boxed is tried first
                        boxed_match_priority=0,
                        try_extract_without_anchor=False,
                    )
                ],
                extraction_mode="first_match",
            )
            # Compute binary rewards if verifiable, `None` otherwise to skip this example
            try:
                reward = float(verify(gold_parsed, answer_parsed))
            except Exception as e:
                print(f"verify failed: {e}, answer: {answer_parsed}, gold: {gold_parsed}")
                reward = None
        else:
            # If the gold solution is not parseable, we assign `None` to skip this example
            reward = None
            print("Failed to parse gold solution: ", sol)
        rewards.append(reward)

    return rewards


def format_reward(completions, **kwargs):
    """Reward function that checks if the reasoning process is enclosed within <think> and </think> tags, while the final answer is enclosed within <answer> and </answer> tags."""
    pattern = r"^<think>\n.*?\n</think>\n<answer>\n.*?\n</answer>$"
    completion_contents = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, content, re.DOTALL | re.MULTILINE) for content in completion_contents]
    return [1.0 if match else 0.0 for match in matches]


def tag_count_reward(completions, **kwargs) -> list[float]:
    """Reward function that checks if we produce the desired number of think and answer tags associated with `format_reward()`.

    Adapted from: https://gist.github.com/willccbb/4676755236bb08cab5f4e54a0475d6fb#file-grpo_demo-py-L90
    """    
    def count_tags(text: str) -> float:
        count = 0.0
        if text.count("<item_nm>") == 1:
            count += 0.25
        if text.count("</item_nm>") == 1:
            count += 0.25
        if text.count("<item_id>") == 1:
            count += 0.25
        if text.count("</item_id>") == 1:
            count += 0.25
        if text.count("<think>") == 1:
            count += 0.25
        if text.count("</think>") == 1:
            count += 0.25
        return count

    contents = [completion[0]["content"] for completion in completions]
    return [count_tags(c) for c in contents]


def reasoning_steps_reward(completions, **kwargs):
    r"""Reward function that checks for clear step-by-step reasoning.
    Regex pattern:
        Step \d+: - matches "Step 1:", "Step 2:", etc.
        ^\d+\. - matches numbered lists like "1.", "2.", etc. at start of line
        \n- - matches bullet points with hyphens
        \n\* - matches bullet points with asterisks
        First,|Second,|Next,|Finally, - matches transition words
    """
    pattern = r"(Step \d+:|^\d+\.|\n-|\n\*|First,|Second,|Next,|Finally,)"
    completion_contents = [completion[0]["content"] for completion in completions]
    matches = [len(re.findall(pattern, content)) for content in completion_contents]

    # Magic number 3 to encourage 3 steps and more, otherwise partial reward
    return [min(1.0, count / 3) for count in matches]


def len_reward(completions: list[Dict[str, str]], solution: list[str], **kwargs) -> float:
    """Compute length-based rewards to discourage overthinking and promote token efficiency.

    Taken from the Kimi 1.5 tech report: https://arxiv.org/abs/2501.12599

    Args:
        completions: List of model completions
        solution: List of ground truth solutions

    Returns:
        List of rewards where:
        - For correct answers: reward = 0.5 - (len - min_len)/(max_len - min_len)
        - For incorrect answers: reward = min(0, 0.5 - (len - min_len)/(max_len - min_len))
    """
    contents = [completion[0]["content"] for completion in completions]

    # First check correctness of answers
    correctness = []
    for content, sol in zip(contents, solution):
        gold_parsed = parse(
            sol,
            extraction_mode="first_match",
            extraction_config=[LatexExtractionConfig()],
        )
        if len(gold_parsed) == 0:
            # Skip unparseable examples
            correctness.append(True)  # Treat as correct to avoid penalizing
            print("Failed to parse gold solution: ", sol)
            continue

        answer_parsed = parse(
            content,
            extraction_config=[
                LatexExtractionConfig(
                    normalization_config=NormalizationConfig(
                        nits=False,
                        malformed_operators=False,
                        basic_latex=True,
                        equations=True,
                        boxed=True,
                        units=True,
                    ),
                    boxed_match_priority=0,
                    try_extract_without_anchor=False,
                )
            ],
            extraction_mode="first_match",
        )
        correctness.append(verify(answer_parsed, gold_parsed))

    # Calculate lengths
    lengths = [len(content) for content in contents]
    min_len = min(lengths)
    max_len = max(lengths)

    # If all responses have the same length, return zero rewards
    if max_len == min_len:
        return [0.0] * len(completions)

    rewards = []
    for length, is_correct in zip(lengths, correctness):
        lambda_val = 0.5 - (length - min_len) / (max_len - min_len)

        if is_correct:
            reward = lambda_val
        else:
            reward = min(0, lambda_val)

        rewards.append(float(reward))

    return rewards


def get_cosine_scaled_reward(
    min_value_wrong: float = -1.0,
    max_value_wrong: float = -0.5,
    min_value_correct: float = 0.5,
    max_value_correct: float = 1.0,
    max_len: int = 1000,
):
    def cosine_scaled_reward(completions, solution, **kwargs):
        """Reward function that scales based on completion length using a cosine schedule.

        Shorter correct solutions are rewarded more than longer ones.
        Longer incorrect solutions are penalized less than shorter ones.

        Args:
            completions: List of model completions
            solution: List of ground truth solutions

        This function is parameterized by the following arguments:
            min_value_wrong: Minimum reward for wrong answers
            max_value_wrong: Maximum reward for wrong answers
            min_value_correct: Minimum reward for correct answers
            max_value_correct: Maximum reward for correct answers
            max_len: Maximum length for scaling
        """
        contents = [completion[0]["content"] for completion in completions]
        rewards = []

        for content, sol in zip(contents, solution):
            gold_parsed = parse(sol, extraction_mode="first_match", extraction_config=[LatexExtractionConfig()])
            if len(gold_parsed) == 0:
                rewards.append(1.0)  # Skip unparseable examples
                print("Failed to parse gold solution: ", sol)
                continue

            answer_parsed = parse(
                content,
                extraction_config=[
                    LatexExtractionConfig(
                        normalization_config=NormalizationConfig(
                            nits=False,
                            malformed_operators=False,
                            basic_latex=True,
                            equations=True,
                            boxed=True,
                            units=True,
                        ),
                        boxed_match_priority=0,
                        try_extract_without_anchor=False,
                    )
                ],
                extraction_mode="first_match",
            )

            is_correct = verify(answer_parsed, gold_parsed)
            gen_len = len(content)

            # Apply cosine scaling based on length
            progress = gen_len / max_len
            cosine = math.cos(progress * math.pi)

            if is_correct:
                min_value = min_value_correct
                max_value = max_value_correct
            else:
                # Swap min/max for incorrect answers
                min_value = max_value_wrong
                max_value = min_value_wrong

            reward = min_value + 0.5 * (max_value - min_value) * (1.0 + cosine)
            rewards.append(float(reward))

        return rewards

    return cosine_scaled_reward


def get_repetition_penalty_reward(ngram_size: int, max_penalty: float):
    """
    Computes N-gram repetition penalty as described in Appendix C.2 of https://arxiv.org/abs/2502.03373.
    Reference implementation from: https://github.com/eddycmu/demystify-long-cot/blob/release/openrlhf/openrlhf/reward/repetition.py

    Args:
    ngram_size: size of the n-grams
    max_penalty: Maximum (negative) penalty for wrong answers
    """
    if max_penalty > 0:
        raise ValueError(f"max_penalty {max_penalty} should not be positive")

    def zipngram(text: str, ngram_size: int):
        words = text.lower().split()
        return zip(*[words[i:] for i in range(ngram_size)])

    def repetition_penalty_reward(completions, **kwargs) -> float:
        """
        reward function the penalizes repetitions
        ref implementation: https://github.com/eddycmu/demystify-long-cot/blob/release/openrlhf/openrlhf/reward/repetition.py

        Args:
            completions: List of model completions
        """

        contents = [completion[0]["content"] for completion in completions]
        rewards = []
        for completion in contents:
            if completion == "":
                rewards.append(0.0)
                continue
            if len(completion.split()) < ngram_size:
                rewards.append(0.0)
                continue

            ngrams = set()
            total = 0
            for ng in zipngram(completion, ngram_size):
                ngrams.add(ng)
                total += 1

            scaling = 1 - len(ngrams) / total
            reward = scaling * max_penalty
            rewards.append(reward)
        return rewards

    return repetition_penalty_reward


def _init_event_loop():
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    return loop


def get_code_format_reward(language: str = "python"):
    """Format reward function specifically for code responses.

    Args:
        language: Programming language supported by E2B https://e2b.dev/docs/code-interpreting/supported-languages
    """
    pattern = rf"^<think>\n.*?\n</think>\n<answer>\n.*?```{language}.*?```.*?\n</answer>$"

    def code_format_reward(completions, **kwargs):
        completion_contents = [completion[0]["content"] for completion in completions]
        matches = [re.match(pattern, content, re.DOTALL | re.MULTILINE) for content in completion_contents]
        return [1.0 if match else 0.0 for match in matches]

    return code_format_reward


def run_async_from_sync(scripts: list[str], language: str, num_parallel: int) -> list[float]:
    """Function wrapping the `run_async` function."""
    # Create a new event loop and set it
    try:
        # Run the async function and get the result
        rewards = asyncio.run(run_async(scripts, language, num_parallel))
    except Exception as e:
        print(f"Error from E2B executor async: {e}")
        raise e

    return rewards


async def run_async(scripts: list[str], language: str, num_parallel: int) -> list[float]:
    # Limit the number of concurrent tasks
    semaphore = asyncio.Semaphore(num_parallel)

    # Create a list of tasks for running scripts concurrently
    tasks = [run_script(script, language, semaphore) for script in scripts]

    # Wait for all tasks to complete and gather their results as they finish
    results = await asyncio.gather(*tasks)
    rewards = list(results)  # collect results

    return rewards


async def run_script(script: str, language: str, semaphore: asyncio.Semaphore) -> float:
    # We set a timeout margin, as the AsyncSandbox timeout does not seem to work
    # These values are based on running 256 examples with the gold solution
    # from open-r1/verifiable-coding-problems-python_decontaminated
    # see scripts/benchmark_e2b.py

    SANDBOX_TIMEOUT = 30
    MARGIN = 2
    REQUEST_TIMEOUT = SANDBOX_TIMEOUT - MARGIN
    ASYNCIO_TIMEOUT = SANDBOX_TIMEOUT + MARGIN

    async with semaphore:
        try:
            sandbox = await AsyncSandbox.create(timeout=SANDBOX_TIMEOUT, request_timeout=REQUEST_TIMEOUT)
            execution = await asyncio.wait_for(sandbox.run_code(script, language=language), timeout=ASYNCIO_TIMEOUT)
            return float(execution.text)
        except (TypeError, ValueError):
            return 0.0
        except asyncio.TimeoutError:
            print("Operation timed out")
            return 0.0
        except Exception as e:
            print(f"Error in `run_script` from E2B sandbox ID {sandbox.sandbox_id} : {e}")
            return 0.0
        finally:
            try:
                await sandbox.kill()
            except Exception as e:
                print(f"Error from E2B executor kill with sandbox ID {sandbox.sandbox_id} : {e}")


def get_reward_funcs(script_args) -> list[Callable]:
    REWARD_FUNCS_REGISTRY = {
        "tag_count": tag_count_reward,
        "item_accuracy" : item_match_reward,
        "item_id_accuracy" : item_id_match_reward,
        "tag_presence" : tag_presence_reward,
        "only_expected_tags" : only_expected_tags_reward,
        "cf_model_reward" : collaborative_filtering_reward,
        "collaborative_guided_reward" : collaborative_guided_reward,
        "conditional_sc_reward" : update_wrapper(
            partial(
                conditional_sc_reward
            ),
            conditional_sc_reward,
        ),
        "embedding_similarity": embedding_similarity_reward,
    }
    reward_funcs = [REWARD_FUNCS_REGISTRY[func] for func in script_args.reward_funcs]

    return reward_funcs

