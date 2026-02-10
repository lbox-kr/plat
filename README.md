# PLAT: Predicting the Legitimacy of punitive Additional Tax

This is the official repository for the **PLAT (Predicting the Legitimacy of punitive Additional Tax)** dataset from [LBox](https://lbox.kr/v2).

**Paper**: [Taxation Perspectives from Large Language Models: A Case Study on Additional Tax Penalties](https://arxiv.org/abs/2503.03444)

## Hugging Face Datasets

The dataset is available on Hugging Face Hub in multiple formats. 

**Collection**: [PLAT: Perspectives on Law And Taxation](https://huggingface.co/collections/sma1-rmarud/plat-perspectives-on-law-and-taxation-698820f928d2264727b8630f)

### Korean Version
| Dataset | Description | Link |
|---------|-------------|------|
| **plat-kor-mc** | Binary (lawful/unlawful) questions | [Link](https://huggingface.co/datasets/sma1-rmarud/plat-kor-mc) |
| **plat-kor-mc4** | 4-choice multiple choice questions | [Link](https://huggingface.co/datasets/sma1-rmarud/plat-kor-mc4) |
| **plat-kor-essay** | Essay-type questions with rubrics | [Link](https://huggingface.co/datasets/sma1-rmarud/plat-kor-essay) |

### English Version
| Dataset | Description | Link |
|---------|-------------|------|
| **plat-eng-mc** | Binary (lawful/unlawful) questions | [Link](https://huggingface.co/datasets/sma1-rmarud/plat-eng-mc) |
| **plat-eng-mc4** | 4-choice multiple choice questions | [Link](https://huggingface.co/datasets/sma1-rmarud/plat-eng-mc4) |
| **plat-eng-essay** | Essay-type questions with rubrics | [Link](https://huggingface.co/datasets/sma1-rmarud/plat-eng-essay) |

## Dataset Overview

PLAT is a benchmark dataset for evaluating legal reasoning in tax law cases. It consists of 100 high-quality Korean tax precedents related to additional tax penalties, translated into English.

### Task Types

1. **Multiple Choice - Binary** (`mc`): Lawful/Unlawful classification
   - Fields: `case_no`, `case_info`, `facts`, `claims`, `reasoning`, `decision`, `lawfulness`

2. **Multiple Choice - 4 Options** (`mc4`): 4-choice questions
   - Fields: `case_no`, `case_info`, `facts`, `claims`, `reasoning`, `decision`, `choices`, `gt`

3. **Essay** (`essay`): Open-ended questions requiring detailed legal analysis
   - Fields: `case_no`, `question_prefix`, `case_info`, `facts`, `claims`, `reasoning`, `decision`, `rubric`

## Installation

### Requirements

```bash
pip install -r requirements.txt
```

### Environment Setup

Create a `.env` file in the project directory:

```bash
OPENAI_API_KEY=sk-your-api-key-here
```

## Quick Start

### Load Dataset

```python
from datasets import load_dataset

# Korean - MC (Binary)
mc_kor = load_dataset("sma1-rmarud/plat-kor-mc")
print(mc_kor["test"][0])

# Korean - MC4 (4-choice)
mc4_kor = load_dataset("sma1-rmarud/plat-kor-mc4")
print(mc4_kor["test"][0])

# Korean - Essay
essay_kor = load_dataset("sma1-rmarud/plat-kor-essay")
print(essay_kor["test"][0])

# English - MC (Binary)
mc_eng = load_dataset("sma1-rmarud/plat-eng-mc")
print(mc_eng["test"][0])

# English - MC4 (4-choice)
mc4_eng = load_dataset("sma1-rmarud/plat-eng-mc4")
print(mc4_eng["test"][0])

# English - Essay
essay_eng = load_dataset("sma1-rmarud/plat-eng-essay")
print(essay_eng["test"][0])
```

### Evaluate with LLM

See `eval.py` for evaluation code examples.

#### CLI Arguments

| Argument | Description | Default | Choices |
|----------|-------------|---------|---------|
| `--task` | Evaluation task | `all` | `mc`, `mc4`, `essay`, `all` |
| `--model` | Model name | `gpt-4o-2024-11-20` | Any model name |
| `--judge-model` | Judge model for essay | `o3-2025-04-16` | Any model name |
| `--lang` | Language | `kor` | `kor`, `eng`, `both` |
| `--num-samples` | Number of samples | `None` (all) | Integer |
| `--output-dir` | Output directory | `./eval_results` | Path |
| `--base-url` | Custom API URL | `None` | URL (for vLLM, etc.) |
| `--api-key` | Custom API key | `None` | API key |

> **Note**: This script uses the OpenAI API format. You can evaluate **any model** (Gemini, Claude, Llama, Qwen, Mistral, etc.) by serving it with an OpenAI-compatible server (e.g., vLLM, TGI, Ollama) and using `--base-url` and `--model` arguments.

#### Usage Examples

```bash
# Run all tasks (mc, mc4, essay) in Korean
python eval.py

# Run only MC4 task
python eval.py --task mc4 --lang kor

# Quick test with 10 samples
python eval.py --task mc --num-samples 10

# Use a different model
python eval.py --model gpt-4o-mini --task mc4

# Use vLLM server
python eval.py --base-url http://localhost:8000/v1 --api-key dummy --model your-model-name

# Run both Korean and English
python eval.py --task mc4 --lang both
```

#### Output

Results are saved to `./eval_results/{task}/`:
- `{model}_{lang}.json`: Detailed results with prompts and responses
- `{model}_{lang}_results.csv`: Summary metrics (accuracy/score)

## Repository Structure

```
plat/
├── README.md           # This file
├── requirements.txt    # Python dependencies
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

## License

CC BY-NC-SA 4.0
