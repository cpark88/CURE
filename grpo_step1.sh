export VLLM_USE_V1="0"
# CUDA_VISIBLE_DEVICES=0 trl vllm-serve --model model/google_gemma-3-4b-it_sft_001_amazon_cdsr_only_single_lora --tensor-parallel-size 1 --max-model-len 13000 --dtype bfloat16 --port 8001
CUDA_VISIBLE_DEVICES=0 trl vllm-serve --model model/google_gemma-3-4b-it_sft_001_amazon_cdsr_only_single_lora --tensor-parallel-size 1 --max-model-len 13000 --dtype bfloat16 --port 8001
