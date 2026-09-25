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

import logging
import os
import sys

import datasets
import torch
import transformers
from datasets import load_dataset
from transformers import set_seed
from transformers.trainer_utils import get_last_checkpoint

from src.open_r1.configs import GRPOConfig, GRPOScriptArguments
from src.open_r1.rewards import get_reward_funcs
from src.open_r1.utils import get_tokenizer, get_model
from src.open_r1.utils.callbacks import get_callbacks
from src.open_r1.utils.wandb_logging import init_wandb_training
from trl import ModelConfig, TrlParser, get_peft_config
# from src.open_r1.utils.edited_grpo_trainer import GRPOEditedTrainer
import json
from datasets import load_dataset, load_from_disk, concatenate_datasets
from datasets import Dataset, DatasetDict
import random


# from src.open_r1.intuitor_trainer import INTUITORTrainer
from src.open_r1.onerec_trainer import INTUITORTrainer

logger = logging.getLogger(__name__)

os.environ['VLLM_USE_V1'] = '0'



def sample_and_filter_by_user(in_domain_data, cross_domain_data, 
                              n_users=1000, seed=11, min_score=4):
    """
    in_domain_data / cross_domain_data: list[dict]
    Each dict is assumed to have 'user_id' and 'score' keys.
    
    Steps:
    - Keep user_ids with score >= min_score in both datasets
    - Sample n_users among those user_ids
    - Return both lists filtered to the sampled user_ids
    """

    if seed is not None:
        random.seed(seed)

    # 1) Collect qualifying user_ids from in-domain data
    valid_in_users = {
        item['user_id']
        for item in in_domain_data
        if item.get('score', 0) >= min_score
    }

    # 2) Collect qualifying user_ids from cross-domain data
    valid_cross_users = {
        item['user_id']
        for item in cross_domain_data
        if item.get('score', 0) >= min_score
    }

    # 3) Keep user_ids that qualify in both datasets
    valid_users = valid_in_users.intersection(valid_cross_users)

    # 4) Cap the number of users to sample
    n_users = min(n_users, len(valid_users))

    # 5) Randomly sample user_ids
    sampled_user_ids = set(random.sample(list(valid_users), n_users))

    # 6) Filter the data
    in_domain_filtered = [
        item for item in in_domain_data
        if item['user_id'] in sampled_user_ids
    ]

    cross_domain_filtered = [
        item for item in cross_domain_data
        if item['user_id'] in sampled_user_ids
    ]

    return in_domain_filtered, cross_domain_filtered, sampled_user_ids


def main(script_args, training_args, model_args):
    # os.environ["WANDB_MODE"] = "offline"
    
    # Set seed for reproducibility
    set_seed(training_args.seed)

    ###############
    # Setup logging
    ###############
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process a small summary
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f" distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Model parameters {model_args}")
    logger.info(f"Script parameters {script_args}")
    logger.info(f"Training parameters {training_args}")

    # Check for last checkpoint
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
    if last_checkpoint is not None and training_args.resume_from_checkpoint is None:
        logger.info(f"Checkpoint detected, resuming training at {last_checkpoint=}.")

    if "wandb" in training_args.report_to:
        init_wandb_training(training_args)

    # Load the dataset
    domain = training_args.domain
    # domain_other = 'Grocery_and_Gourmet_Food'
    
    
    
    
    dataset_train = []
    n_users = 500

    with open(f"./data/Amazon_Fashion/llm_dataset/amazon_Amazon_Fashion_Amazon_Fashion_llm_train_case_1_20251102.json", 'r', encoding='utf-8') as f:
        dataset_train_fashion_fashion = json.load(f)
        # random.shuffle(dataset_train_fashion_fashion)
        print("train1:", len(dataset_train_fashion_fashion))
        # dataset_train.extend(dataset_train_fashion_fashion)

    with open(f"./data/Amazon_Fashion/llm_dataset/amazon_Amazon_Fashion_Clothing_Shoes_and_Jewelry_llm_train_case_1_20251102.json", 'r', encoding='utf-8') as f:
        dataset_train_fashion_clothing = json.load(f)
        # random.shuffle(dataset_train_fashion_clothing)
        print("train2:", len(dataset_train_fashion_clothing))
        # dataset_train.extend(dataset_train_fashion_clothing)
        
        
    dataset_train_fashion_fashion_filtered, dataset_train_fashion_clothing_filtered, _ = sample_and_filter_by_user(in_domain_data=dataset_train_fashion_fashion, cross_domain_data=dataset_train_fashion_clothing, n_users=n_users, seed=11)
    dataset_train.extend(dataset_train_fashion_fashion_filtered)
    dataset_train.extend(dataset_train_fashion_clothing_filtered)
        
        
    with open(f"./data/Clothing_Shoes_and_Jewelry/llm_dataset/amazon_Clothing_Shoes_and_Jewelry_Clothing_Shoes_and_Jewelry_llm_train_case_1_20251102.json", 'r', encoding='utf-8') as f:
        dataset_train_clothing_clothing = json.load(f)
        # random.shuffle(dataset_train_clothing_clothing)
        print("train3:", len(dataset_train_clothing_clothing))
        # dataset_train.extend(dataset_train_clothing_clothing)

    with open(f"./data/Clothing_Shoes_and_Jewelry/llm_dataset/amazon_Clothing_Shoes_and_Jewelry_Amazon_Fashion_llm_train_case_1_20251102.json", 'r', encoding='utf-8') as f:
        dataset_train_clothing_fashion = json.load(f)
        # random.shuffle(dataset_train_clothing_fashion)
        print("train4:", len(dataset_train_clothing_fashion))
        # dataset_train.extend(dataset_train_clothing_fashion)

    dataset_train_clothing_clothing_filtered, dataset_train_clothing_fashion_filtered, _ = sample_and_filter_by_user(in_domain_data=dataset_train_clothing_clothing, cross_domain_data=dataset_train_clothing_fashion, n_users=n_users, seed=11)
    dataset_train.extend(dataset_train_clothing_clothing_filtered)
    dataset_train.extend(dataset_train_clothing_fashion_filtered)        
        
    with open(f"./data/Grocery_and_Gourmet_Food/llm_dataset/amazon_Grocery_and_Gourmet_Food_Grocery_and_Gourmet_Food_llm_train_case_1_20251102.json", 'r', encoding='utf-8') as f:
        dataset_train_grocery_grocery = json.load(f)
        # random.shuffle(dataset_train_grocery_grocery)
        print("train5:", len(dataset_train_grocery_grocery))
        # dataset_train.extend(dataset_train_grocery_grocery)
        
    with open(f"./data/Grocery_and_Gourmet_Food/llm_dataset/amazon_Grocery_and_Gourmet_Food_Health_and_Household_llm_train_case_1_20251102.json", 'r', encoding='utf-8') as f:
        dataset_train_grocery_health = json.load(f)
        # random.shuffle(dataset_train_grocery_health)
        print("train6:", len(dataset_train_grocery_health))
        # dataset_train.extend(dataset_train_grocery_health)

        
    dataset_train_grocery_grocery_filtered, dataset_train_grocery_health_filtered, _ = sample_and_filter_by_user(in_domain_data=dataset_train_grocery_grocery, cross_domain_data=dataset_train_grocery_health, n_users=n_users, seed=11)
    dataset_train.extend(dataset_train_grocery_grocery_filtered)
    dataset_train.extend(dataset_train_grocery_health_filtered)         
        
        
        
    with open(f"./data/Health_and_Household/llm_dataset/amazon_Health_and_Household_Health_and_Household_llm_train_case_1_20251102.json", 'r', encoding='utf-8') as f:
        dataset_train_health_health = json.load(f)
        # random.shuffle(dataset_train_health_health)
        print("train7:", len(dataset_train_health_health))
        # dataset_train.extend(dataset_train_health_health)
        
    with open(f"./data/Health_and_Household/llm_dataset/amazon_Health_and_Household_Grocery_and_Gourmet_Food_llm_train_case_1_20251102.json", 'r', encoding='utf-8') as f:
        dataset_train_health_grocery = json.load(f)
        # random.shuffle(dataset_train_health_grocery)
        print("train8:", len(dataset_train_health_grocery))
        # dataset_train.extend(dataset_train_health_grocery)
        
    dataset_train_health_health_filtered, dataset_train_health_grocery_filtered, _ = sample_and_filter_by_user(in_domain_data=dataset_train_health_health, cross_domain_data=dataset_train_health_grocery, n_users=n_users, seed=11)
    dataset_train.extend(dataset_train_health_health_filtered)
    dataset_train.extend(dataset_train_health_grocery_filtered)        
        
        
        
    with open(f"./data/ml_32m/llm_dataset/amazon_ml_32m_ml_32m_llm_train_case_1_20251102.json", 'r', encoding='utf-8') as f:
        dataset_train_ml_ml = json.load(f)
        # random.shuffle(dataset_train_ml_ml)
        print("train9:", len(dataset_train_ml_ml))
        # dataset_train.extend(dataset_train_ml_ml)
        
    with open(f"./data/ml_32m/llm_dataset/amazon_ml_32m_CDs_and_Vinyl_llm_train_case_1_20251102.json", 'r', encoding='utf-8') as f:
        dataset_train_ml_cd = json.load(f)
        # random.shuffle(dataset_train_ml_cd)
        print("train10:", len(dataset_train_ml_cd))
        # dataset_train.extend(dataset_train_ml_cd)
        
    dataset_train_ml_ml_filtered, dataset_train_ml_cd_filtered, _ = sample_and_filter_by_user(in_domain_data=dataset_train_ml_ml, cross_domain_data=dataset_train_ml_cd, n_users=n_users, seed=11)
    dataset_train.extend(dataset_train_ml_ml_filtered)
    dataset_train.extend(dataset_train_ml_cd_filtered)        
        
        
    with open(f"./data/CDs_and_Vinyl/llm_dataset/amazon_CDs_and_Vinyl_CDs_and_Vinyl_llm_train_case_1_20251102.json", 'r', encoding='utf-8') as f:
        dataset_train_cd_cd = json.load(f)
        # random.shuffle(dataset_train_cd_cd)
        print("train11:", len(dataset_train_cd_cd))
        # dataset_train.extend(dataset_train_cd_cd)
        
    with open(f"./data/CDs_and_Vinyl/llm_dataset/amazon_CDs_and_Vinyl_ml_32m_llm_train_case_1_20251102.json", 'r', encoding='utf-8') as f:
        dataset_train_cd_ml = json.load(f)
        # random.shuffle(dataset_train_cd_ml)
        print("train12:", len(dataset_train_cd_ml))
        # dataset_train.extend(dataset_train_cd_ml)
        
        
    dataset_train_cd_cd_filtered, dataset_train_cd_ml_filtered, _ = sample_and_filter_by_user(in_domain_data=dataset_train_cd_cd, cross_domain_data=dataset_train_cd_ml, n_users=n_users, seed=11)
    dataset_train.extend(dataset_train_cd_cd_filtered)
    dataset_train.extend(dataset_train_cd_ml_filtered)         
        
        
    print("total data:", len(dataset_train))
            
    for record in dataset_train:
        record["score"] = str(record["score"])
        record["positive_item_id"] = str(record["positive_item_id"])
        record["user_id"] = str(record["user_id"])
    
        

    # with open(dataset_name, 'r', encoding='utf-8') as f:
    #     dataset_test = json.load(f)
    # random.shuffle(dataset_test)
    # dataset_test = dataset_test[:500]  
    
    # random.shuffle(dataset_train)
    # dataset_train = Dataset.from_list(dataset_train[:10000])
    dataset_train = Dataset.from_list(dataset_train)
    dataset_test = dataset_train #Dataset.from_list(dataset_train[:500])
    
    dataset = DatasetDict({
                "train": dataset_train,
                "test": dataset_test})

    ################
    # Load tokenizer
    ################
    tokenizer = get_tokenizer(model_args, training_args)

    ##############
    # Load model #
    ##############
    logger.info("*** Loading model ***")
    model = get_model(model_args, training_args)
    
    # Get reward functions from the registry
    reward_funcs = get_reward_funcs(script_args)

    # Format into conversation
    def make_conversation(example):
        prompt = []

        prompt.append({"role": "system", "content": example["system_prompt"]})
        prompt.append({"role": "user", "content": example["input"]})
        return {"prompt": prompt}

    data_seed = 42
    dataset['train'] = dataset['train'].shuffle(data_seed).map(make_conversation)
    dataset['test'] = dataset['test'].shuffle(data_seed).map(make_conversation)

    for split in dataset:
        if "messages" in dataset[split].column_names:
            dataset[split] = dataset[split].remove_columns("messages")

    #############################
    # Initialize the GRPO trainer
    #############################    
    trainer = INTUITORTrainer(
        model=model_args.model_name_or_path,
        reward_funcs=reward_funcs,
        args=training_args,
        train_dataset=dataset[script_args.dataset_train_split],
        eval_dataset=(dataset[script_args.dataset_test_split] if training_args.eval_strategy != "no" else None),
        peft_config=get_peft_config(model_args),
        callbacks=get_callbacks(training_args, model_args),
        processing_class=tokenizer,
    )

    ###############
    # Training loop
    ###############
    logger.info("*** Train ***")
    checkpoint = None
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
    elif last_checkpoint is not None:
        checkpoint = last_checkpoint
    train_result = trainer.train(resume_from_checkpoint=checkpoint)
    metrics = train_result.metrics
    metrics["train_samples"] = len(dataset[script_args.dataset_train_split])
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    ##################################
    # Save model and create model card
    ##################################
    logger.info("*** Save model ***")
    trainer.save_model(training_args.output_dir)
    logger.info(f"Model saved to {training_args.output_dir}")

    # Save everything else on main process
    kwargs = {
        "dataset_name": script_args.dataset_name,
        "tags": ["open-r1"],
    }
    if trainer.accelerator.is_main_process:
        trainer.create_model_card(**kwargs)
        trainer.model.config.save_pretrained(training_args.output_dir)

    ##########
    # Evaluate
    ##########
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate()
        metrics["eval_samples"] = len(dataset[script_args.dataset_test_split])
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    #############
    # push to hub
    #############
    if training_args.push_to_hub:
        logger.info("Pushing to hub...")
        trainer.push_to_hub(**kwargs)


if __name__ == "__main__":
    parser = TrlParser((GRPOScriptArguments, GRPOConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    
    main(script_args, training_args, model_args)
