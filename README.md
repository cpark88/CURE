<div align="center">

# CURE (NeurIPS 2026)

### Coupled User-Grouped Reinforcement Learning for Cross-Domain Recommendation with Non-Overlapping Users

[![NeurIPS 2026](https://img.shields.io/badge/NeurIPS-2026-8A2BE2)](https://neurips.cc/Conferences/2026)
[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.6-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Backbone](https://img.shields.io/badge/Backbone-Gemma--3--4B--it-4285F4)](https://huggingface.co/google/gemma-3-4b-it)
[![Dataset](https://img.shields.io/badge/Data-Google%20Drive-34A853?logo=googledrive&logoColor=white)](https://drive.google.com/drive/folders/1uRyKuDbfqsOvr31d1UzJUoLm2SThp3eA?usp=drive_link)

</div>

---

## 📢 News

- **[2026.09]** CURE has been accepted to **NeurIPS 2026**! 🎉

---

## Overview

This is the official implementation of **CURE**. Cross-domain recommendation usually assumes users overlap across services, but in practice that overlap is tiny and data-sharing constraints make it even smaller. CURE targets this *non-overlapping user* setting with two components:

- **Agentic Synthetic Data Generation Pipeline (ASP)**: builds cross-domain pseudo supervision from a user's source-domain history, which is otherwise unavailable when users don't overlap.
- **Coupled User-Grouped RL**: jointly normalizes in-domain and cross-domain rewards at the user level, so noisy synthetic cross-domain updates are calibrated against reliable in-domain signals.

### Highlights

- Consistent gains over CF-based and LLM-based baselines on **Amazon Reviews** and **MovieLens**, for both cross-domain and in-domain recommendation.
- Theoretical analysis showing that user-level coupling stabilizes policy updates under imperfect pseudo supervision.
- **+56% CTR** (statistically significant) over existing ML methods in an online A/B test.

<details>
<summary><b>Abstract</b></summary>
<br>

Cross-Domain Recommendation is essential for enabling cross-sell in multi-service platforms; however, limited user overlap and strict cross-service data-sharing constraints often result in little to no ground-truth supervision for cross-domain learning.
To bridge this gap, we propose an LLM-based framework for cross-domain recommendation in the non-overlapping user scenario.
First, we propose an agentic pipeline that constructs cross-domain pseudo supervision, which is unavailable in the non-overlapping user scenario, by leveraging a user’s source-domain history to generate a target-domain item recommendation.
Using this dataset, we further propose **C**oupled **U**ser-G**R**ouped R**E**inforcement Learning (CURE), which induces coupling between in- and cross-domain gradient updates via user-level joint normalization, thereby calibrating synthetic cross-domain updates against reliable in-domain signals.
We provide a theoretical analysis showing that user-level coupling stabilizes policy updates under imperfect pseudo supervision and improves cross-domain generalization.
Experiments on Amazon Reviews and MovieLens show consistent gains over baselines in cross-domain recommendation, while also improving in-domain performance.
An online A/B test demonstrates a statistically significant 56% lift in click-through rate over existing ML methods.

</details>

---

## Repository Structure

```
.
├── sft.sh                         # Stage 1: supervised fine-tuning
├── grpo_step1.sh                  # Stage 2-1: launch vLLM server with the SFT model
├── grpo_step2.sh                  # Stage 2-2: CURE training
├── evaluation_vllm.py             # In-domain evaluation (Amazon)
├── evaluation_vllm_cross.py       # Cross-domain evaluation (Amazon)
├── evaluation_vllm_movie_lens.py  # Evaluation on MovieLens
├── recipes/
│   ├── CURE/                      # SFT / RL training configs
│   └── accelerate_configs/        # DeepSpeed / FSDP / DDP configs
└── src/open_r1/
    ├── sft.py                     # SFT entry point
    ├── intuitor.py                # CURE entry point
    ├── onerec_trainer.py          # CURE trainer (grouped + calibration)
    ├── onerec_only_grouped_trainer.py  # Ablation: grouped, no calibration
    ├── grpo.py / intuitor_trainer.py   # Intuitor baseline
    ├── rewards.py                 # Reward functions
    └── sasrec/                    # SASRec model used for CF-based rewards
```

---

## Getting Started

### 1. Environment

```bash
pip install -r requirements.txt
```

We use Python 3.10+ and `transformers==4.51.3`, `torch==2.6.0`, `trl`, and `vllm`. You'll need access to [`google/gemma-3-4b-it`](https://huggingface.co/google/gemma-3-4b-it) on the Hugging Face Hub; log in with `huggingface-cli login` or set `HF_TOKEN`.

### 2. Dataset

We release all the data used in the paper. It is reconstructed from the Amazon Reviews 2023 corpus, and the recommendation rationales were generated and refined with a large-scale LLM.

📦 **Download:** [Google Drive](https://drive.google.com/drive/folders/1uRyKuDbfqsOvr31d1UzJUoLm2SThp3eA?usp=drive_link)

Put the downloaded files under `data/`.

### 3. Supervised Fine-Tuning

```bash
bash sft.sh
```

This trains a LoRA adapter on top of Gemma-3-4B-it and then merges it into `model/google_gemma-3-4b-it_sft_001_amazon_cdsr_only_single_lora`.

### 4. CURE Training

CURE uses a separate vLLM server for generation, so it runs in two steps.

**Step 1.** Launch the SFT model with vLLM on one GPU:

```bash
bash grpo_step1.sh
```

**Step 2.** Run CURE training on the remaining GPUs:

```bash
bash grpo_step2.sh
```

Hyperparameters are in `recipes/CURE/grpo/grpo_config_amazon.yaml`.

### 5. Evaluation

```bash
# In-domain
python evaluation_vllm.py --domain Amazon_Fashion --num_gpus 2 --stage ours

# Cross-domain
python evaluation_vllm_cross.py --source_domain Amazon_Fashion --target_domain Clothing_Shoes_and_Jewelry --num_gpus 2
```

Set `model_name` inside each script to the checkpoint you want to evaluate.

---

## Citation

If you find this work useful, please cite our paper:

```bibtex
@inproceedings{cure2026,
  title={CURE: Coupled User-Grouped Reinforcement Learning for Cross-Domain Recommendation with Non-Overlapping Users},
  author={TBD},
  booktitle={Advances in Neural Information Processing Systems (NeurIPS)},
  year={2026}
}
```

If you use the data, please also cite the Amazon Reviews 2023 paper:

```bibtex
@article{hou2024bridging,
  title={Bridging Language and Items for Retrieval and Recommendation},
  author={Hou, Yupeng and Li, Jiacheng and He, Zhankui and Yan, An and Chen, Xiusi and McAuley, Julian},
  journal={arXiv preprint arXiv:2403.03952},
  year={2024}
}
```

---

## Acknowledgements

This codebase is built on top of [Open-R1](https://github.com/huggingface/open-r1) and [TRL](https://github.com/huggingface/trl). We thank the authors for making their code available.
