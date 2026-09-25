import multiprocessing as mp
import numpy as np
from datasets import load_from_disk

import torch
import tqdm
import json
import re
import pandas as pd
import sys
import subprocess
import os
import math
from vllm import LLM, SamplingParams
from transformers import AutoModelForCausalLM, AutoTokenizer, LogitsProcessor, LogitsProcessorList, BitsAndBytesConfig
import argparse

os.environ['VLLM_USE_V1'] = '0'



import re




def filter_valid_asins(string_list):
    """
    Args:
        string_list (list of str)

    Returns:
        list of str
    """
    pattern = re.compile(r'^[A-Z0-9]{10}$')
    return [s for s in string_list if pattern.match(s)]

def extract_items(text):
    try:
        items = re.findall(r"<item_id>(.*?)</item_id>", text, re.DOTALL)[0].strip()
    except:
        items = []
    return items if items else '//no'

def extract_items_name(text):
    try:
        items = re.findall(r"<item_nm>(.*?)</item_nm>", text, re.DOTALL)[0].strip()
    except:
        items = []
    return items if items else '//no'

def extract_reasoning(text):
    try:
        items = re.findall(r"<think>(.*?)</think>", text, re.DOTALL)[0].strip()
    except:
        items = []
    return items if items else '//no'


def calculate_hr_ndcg(outputs: list, ground_truth: str, top_n: int = 5):
    """
    Args:
        outputs (list): LLM outputs
        ground_truth (str)
        top_n (int)

    Returns:
        hr (float): Hit Rate@N
        ndcg (float): NDCG@N
    """
    top_outputs = outputs[:top_n]

    if ground_truth in top_outputs:
        hr = 1.0
        rank = top_outputs.index(ground_truth) + 1  # 1-based index
        ndcg = 1 / math.log2(rank + 1)
    else:
        hr = 0.0
        ndcg = 0.0

    return hr, ndcg

def evaluate_all(predictions: list, ground_truths: list, top_n: int = 10):
    """
    predictions: list of list
    ground_truths: list of str
    """
    hr_total, ndcg_total = 0.0, 0.0
    num_samples = len(ground_truths)

    for pred, gt in zip(predictions, ground_truths):
        hr, ndcg = calculate_hr_ndcg(pred, gt, top_n)
        hr_total += hr
        ndcg_total += ndcg

    return hr_total / num_samples, ndcg_total / num_samples


def dedup_preserve_order(seq):
    return list(dict.fromkeys(seq))


def inference():
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default='case1', type=str)
    parser.add_argument("--type_", default='001', type=str)
    parser.add_argument("--stage", default='sft', type=str)
    parser.add_argument("--domain", default='Amazon_Fashion', type=str)
    parser.add_argument("--num_gpus", default=2, type=int)
    args = parser.parse_args()
    
    
    
    task = args.task
    type_ = args.type_
    stage = args.stage
    domain = args.domain
    num_gpus = args.num_gpus
    
    print(domain)
    # model_name = f"model/google_gemma-3-4b-it_{stage}_{type_}_amazon_cdsr_only_single_lora"
    # model_name = f"model/google_gemma-3-4b-it_{stage}_{type_}_amazon_cdsr_only_single_lora"
    # model_name = f"model/google_gemma-3-4b-it_{stage}_{type_}_amazon_cdsr_grouped_reward_no_confidence_lora"
    # model_name = f"model/google_gemma-3-4b-it_{stage}_{type_}_amazon_cdsr_grouped_reward_partial_confidence_lora"
    # model_name = f"model/google_gemma-3-4b-it_{stage}_{type_}_amazon_cdsr_grouped_reward_senbert_lora"
    
    # model_name = f"model/google_gemma-3-4b-it_grpo_001_amazon_{domain}_lora" #grpo
    # model_name = "google/gemma-3-4b-it"
    # model_name = f"model/google_gemma-3-4b-it_dpo_001_amazon_{domain}_lora" #DPO
    model_name = f"model/google_gemma-3-4b-it_grpo_001_amazon_Clothing_Shoes_and_Jewelry_lora" #DPO

    llm = LLM(model=model_name, max_model_len=13000, tensor_parallel_size=num_gpus, dtype=torch.bfloat16, trust_remote_code=True)#torch.bfloat16)

    tokenizer = llm.get_tokenizer()
    
    # with open(f'data/test_dataset/amazon_{domain}_llm_test_20250521.json', 'r', encoding='utf-8') as f:
    #     df_final_amazon = json.load(f) # single domain
    with open(f'data/test_dataset/amazon_{domain}_llm_test_20250521.json', 'r', encoding='utf-8') as f:
        df_final_amazon = json.load(f) # cross doamin
    ground_truth = [ extract_items(df_final_amazon[index]['output']) for index in range(len(df_final_amazon)) ]
        
        
    def make_sft_conversation(example, prompt_column: str = 'input'): # prompt / completion
        prompt = []

        prompt.append({"role": "system", "content": example["system_prompt"]})        
        prompt.append({"role": "user", "content": example[prompt_column]})

        # return prompt #vllm serve
        return {'text':tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)}   


    num_gen = 12 #8
    sampling_params_with_processor = SamplingParams(
        temperature=0.7,
        top_k=50,
        top_p=0.9,
        max_tokens=500,
        n=num_gen,
        stop = ["</item_id>"],
    )
    
    # sampling_params_with_processor = SamplingParams(
    #     temperature=1.2,#0.2,#0.7,#0.1,  # sampling temperature
    #     top_k=100,#50,  # keep only the top-k tokens
    #     top_p=0.95,#0.95,#0.1, # nucleus sampling threshold
    #     # repetition_penalty=1.1,  # discourage repeated tokens
    #     max_tokens=500, #300,
    #     n=num_gen,
    #     stop = ["</item_id>"], # truncated during post-processing anyway
    # )

    rationals_name_ = f"""result/inference_list_{domain}_test_{stage}_ndcg_{type_}_{task}.jsonl"""  
    
    prompts = [make_sft_conversation(x)['text'] for x in df_final_amazon]
    batch_size = 128
    
    
    with open(rationals_name_, 'w', encoding='utf-8') as f:
        rationals = []
        rationals_ = []
        
        # Batched inference loop
        for index in tqdm.tqdm(range(0, len(prompts), batch_size)):
            batch_prompts = prompts[index:index + batch_size]       
            output_text = llm.generate(batch_prompts, sampling_params_with_processor)
            print(output_text[0].outputs[3].text)
            output_text = [ filter_valid_asins (  dedup_preserve_order(list(set( [ extract_items(x.outputs[k].text+"</item_id>") for k in range(num_gen)] ))) ) for x in output_text]
            
            for j, output in enumerate(output_text):
                record = {str(index): output}
                rationals.append(record)
                f.write(json.dumps(record, ensure_ascii=False) + '\n')
            
            rationals_.extend(output_text)
                  
    # save
    rationals_name = f"""result/inference_list_{domain}_test_{stage}_ndcg_{type_}_{task}.json"""
    with open(rationals_name, "w") as f:
        json.dump(rationals, f)  # JSON

        
    print("="*100)
    print(domain, 'with ',stage)
    k_=5    
    hr_at_k, ndcg_at_k = evaluate_all(rationals_, ground_truth, top_n=k_)
    print(f"HR@{k_}: {hr_at_k:.4f}, NDCG@{k_}: {ndcg_at_k:.4f}")  
    
    k_=1   
    hr_at_k, ndcg_at_k = evaluate_all(rationals_, ground_truth, top_n=k_)
    print(f"HR@{k_}: {hr_at_k:.4f}, NDCG@{k_}: {ndcg_at_k:.4f}")  
        
        
        

    
    
if __name__ == "__main__":
    inference()