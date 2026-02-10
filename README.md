# PLAT: Predicting the Legitimacy of punitive Additional Tax

This is the official repository for the **PLAT (Predicting the Legitimacy of punitive Additional Tax)** dataset from [LBox](https://lbox.kr/v2).

📄 **Paper**: [Taxation Perspectives from Large Language Models: A Case Study on Additional Tax Penalties](https://arxiv.org/abs/2503.03444)

## Hugging Face Datasets

The dataset is available on Hugging Face Hub in multiple formats.

**Collection**: [PLAT: Perspectives on Law And Taxation](https://huggingface.co/collections/sma1-rmarud/plat-perspectives-on-law-and-taxation-698820f928d2264727b8630f)

### English Version
| Dataset | Description | Link |
|---------|-------------|------|
| **plat-eng-essay** | Essay-type questions with rubrics | [🔗 Link](https://huggingface.co/datasets/sma1-rmarud/plat-eng-essay) |
| **plat-eng-mc** | Binary (lawful/unlawful) questions | [🔗 Link](https://huggingface.co/datasets/sma1-rmarud/plat-eng-mc) |
| **plat-eng-mc4** | 4-choice multiple choice questions | [🔗 Link](https://huggingface.co/datasets/sma1-rmarud/plat-eng-mc4) |

### Korean Version
| Dataset | Description | Link |
|---------|-------------|------|
| **plat-kor-essay** | Essay-type questions with rubrics | [🔗 Link](https://huggingface.co/datasets/sma1-rmarud/plat-kor-essay) |
| **plat-kor-mc** | Binary (lawful/unlawful) questions | [🔗 Link](https://huggingface.co/datasets/sma1-rmarud/plat-kor-mc) |
| **plat-kor-mc4** | 4-choice multiple choice questions | [🔗 Link](https://huggingface.co/datasets/sma1-rmarud/plat-kor-mc4) |

## Dataset Overview

PLAT is a benchmark dataset for evaluating legal reasoning in tax law cases. It consists of 100 high-quality Korean tax precedents related to additional tax penalties, translated into English.

### Task Types

1. **Essay** (`essay`): Open-ended questions requiring detailed legal analysis
   - Fields: `case_no`, `question_prefix`, `case_info`, `facts`, `claims`, `reasoning`, `decision`, `rubric`

2. **Multiple Choice - Binary** (`mc`): Lawful/Unlawful classification
   - Fields: `case_no`, `case_info`, `facts`, `claims`, `reasoning`, `decision`, `lawfulness`

3. **Multiple Choice - 4 Options** (`mc4`): 4-choice questions
   - Fields: `case_no`, `case_info`, `facts`, `claims`, `reasoning`, `decision`, `choices`, `gt`

## Quick Start

### Load Dataset

```python
from datasets import load_dataset

# English - Essay
essay_eng = load_dataset("sma1-rmarud/plat-eng-essay")
print(essay_eng["test"][0])

# English - MC (Binary)
mc_eng = load_dataset("sma1-rmarud/plat-eng-mc")
print(mc_eng["test"][0])

# English - MC4 (4-choice)
mc4_eng = load_dataset("sma1-rmarud/plat-eng-mc4")
print(mc4_eng["test"][0])

# Korean - Essay
essay_kor = load_dataset("sma1-rmarud/plat-kor-essay")
print(essay_kor["test"][0])

# Korean - MC (Binary)
mc_kor = load_dataset("sma1-rmarud/plat-kor-mc")
print(mc_kor["test"][0])

# Korean - MC4 (4-choice)
mc4_kor = load_dataset("sma1-rmarud/plat-kor-mc4")
print(mc4_kor["test"][0])
```

### Evaluate with LLM

See `eval.py` for evaluation code examples.

```bash
python eval.py
```

## Repository Structure

```
plat/
├── README.md           # This file
├── eval.py             # Evaluation script for LLM
└── explore_plat.ipynb  # Jupyter notebook for data exploration
```

## Citation

```bibtex
@misc{choi2026taxationperspectiveslargelanguage,
      title={Taxation Perspectives from Large Language Models: A Case Study on Additional Tax Penalties}, 
      author={Eunkyung Choi and Youngjin Suh and Siun Lee and Hongseok Oh and Juheon Kang and Won Hur and Hun Park and Wonseok Hwang},
      year={2026},
      eprint={2503.03444},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2503.03444}, 
}
```

## 📜 License

CC BY-NC-SA 4.0
