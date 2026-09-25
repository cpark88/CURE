CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 ACCELERATE_LOG_LEVEL=info \
    accelerate launch --config_file recipes/accelerate_configs/zero2_grpo.yaml --num_processes 7 \
    src/open_r1/intuitor.py --config recipes/CURE/grpo/grpo_config_amazon.yaml
    
# python src/open_r1/model_save.py --config recipes/CURE/grpo/grpo_config_amazon.yaml