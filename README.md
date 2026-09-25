# OneRec

## News
***
- **[2026.09]** Our paper has been accepted to **NeurIPS 2026**!

## Overview
***
Official implementation of our NeurIPS 2026 paper "CURE: Coupled User-Grouped Reinforcement Learning for Cross-Domain Recommendation with Non-Overlapping Users".
We referred to the source code of Open-R1 (<https://github.com/huggingface/open-r1>).

## Abstract
***
Cross-Domain Recommendation is essential for enabling cross-sell in multi-service platforms; however, limited user overlap and strict cross-service data-sharing constraints often result in little to no ground-truth supervision for cross-domain learning.
To bridge this gap, we propose an LLM-based framework for cross-domain recommendation in the non-overlapping user scenario.
First, we propose an agentic pipeline that constructs cross-domain pseudo supervision, which is unavailable in the non-overlapping user scenario, by leveraging a user’s source-domain history to generate a target-domain item recommendation. 
Using this dataset, we further propose **C**oupled **U**ser-G**R**ouped R**E**inforcement Learning (CURE), which induces coupling between in- and cross-domain gradient updates via user-level joint normalization, thereby calibrating synthetic cross-domain updates against reliable in-domain signals.
We provide a theoretical analysis showing that user-level coupling stabilizes policy updates under imperfect pseudo supervision and improves cross-domain generalization.
Experiments on Amazon Reviews and MovieLens show consistent gains over baselines in cross-domain recommendation, while also improving in-domain performance.
An online A/B test demonstrates a statistically significant 56% lift in click-through rate over existing ML methods.

## Environment Setting
***
```bash
pip install -r requirements.txt
```


## Dataset
***
We openly release all the data used in our paper. This dataset is reconstructed from the Amazon Reviews 2023 corpus, and the recommendation rationales were generated and refined using a large-scale LLM.

https://drive.google.com/drive/folders/1uRyKuDbfqsOvr31d1UzJUoLm2SThp3eA?usp=drive_link

If you use the data, please also cite the original Amazon Reviews 2023 paper:

```bibtex
@article{hou2024bridging,
  title={Bridging Language and Items for Retrieval and Recommendation},
  author={Hou, Yupeng and Li, Jiacheng and He, Zhankui and Yan, An and Chen, Xiusi and McAuley, Julian},
  journal={arXiv preprint arXiv:2403.03952},
  year={2024}
}
```

## Supervised Finetuning
***

```bash
bash sft.sh
```

## CURE
1. **Start the SFT Model Server (vLLM-based)**

On a machine with a specific GPU, launch the SFT model using `vLLM`:
***

```bash
bash grpo_step1.sh
```

2.	**Run CURE Training**
On a different GPU (or machine), run the GRPO training by executing:

```bash
bash grpo_step2.sh
```

## Citation
***
If you find this work useful, please cite our paper:

```bibtex
@inproceedings{cure2026,
  title={CURE: Coupled User-Grouped Reinforcement Learning for Cross-Domain Recommendation with Non-Overlapping Users},
  author={TBD},
  booktitle={Advances in Neural Information Processing Systems (NeurIPS)},
  year={2026}
}
```
