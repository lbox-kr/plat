"""
PLAT Evaluation Script
======================
Evaluate LLMs on PLAT dataset (Perspectives on Law And Taxation)

Supports:
- MC (Binary): Lawful/Unlawful/Unknown classification (3-choice)
- MC4: 4-choice multiple choice (A/B/C/D)
- Essay: An essay-style legal analysis question evaluated using an LLM-as-a-judge based on a rubric.
"""

import os
import re
import json
import argparse
import logging
import pandas as pd
from datasets import load_dataset
from dotenv import load_dotenv
from tqdm import tqdm
from pathlib import Path
import openai

load_dotenv()

# ============================================================================
# Text Processing Utilities
# ============================================================================

def clean_text(text: str) -> str:
    """Remove markdown and clean text."""
    text = text.replace('\n', ' ')
    text = re.sub(r"[`*_:\[\]\(\)]", "", text)
    return text.strip()


def extract_first_legality(text: str) -> str:
    """
    Extract legality judgment from model response (MC task).
    Returns: "적법하지 않음", "적법함", or "알 수 없음"
    """
    text = text.replace('\n', ' ')
    text = clean_text(text)
    text = re.sub(r"[*_`]+", "", text)

    patterns = [
        ("알 수 없음", r"알\s*수\s*없\s*음|알수없음"),
        ("적법하지 않음", r"적\s*법\s*하\s*지\s*"),
        ("적법함", r"적\s*법\s*함|적법함|적법하다고")
    ]
    for label, pattern in patterns:
        if re.search(pattern, text):
            return label
    return None


def extract_first_choice(text: str) -> str:
    """
    Extract A/B/C/D choice from model response (MC4 task).
    Returns: "A", "B", "C", or "D"
    """
    text = text.replace('\n', ' ')
    pattern = r"답변\s*[:：]?\s*[\*\(“\"'‘’\s]*([A-D])[\*\)”\"'’\s]*"
    match = re.search(pattern, text)
    if match:
        return match.group(1)
    
    # Fallback: find first A/B/C/D
    match = re.search(r'\b([A-D])\b', text)
    if match:
        return match.group(1)
    return None


def parse_point(model_ans: str) -> float:
    """Extract score from essay evaluation response."""
    search = re.search(r"\[\[(\d+)\]\]", model_ans)
    answer = float(search.group(1) if search else 0)
    return answer


# ============================================================================
# Prompts
# ============================================================================

SYSTEM_PROMPT = "당신은 사용자의 질문에 친절하고 논리적으로 답변해 주는 세무 전문가 챗봇입니다."

SYSTEM_PROMPT_ENG = "You are a tax expert chatbot that provides friendly and logical answers to users’ questions."

PROMPT_MC = """위 제시된 가산세 부과처분의 배경을 읽고 해당 가산세 부과에 대해, "적법함", "적법하지 않음" 둘 중 하나를 반드시 선택하여 답변하고 설명해주세요. 답을 내릴 수 없다면 "알수없음"으로 답변하고 설명해주세요.:\n{case_text}"""

PROMPT_MC_ENG = """Please read the background and materials related to the imposed penalty tax presented above. Determine whether the penalty tax is “Legitimate”, “Not legitimate”, or, if a clear conclusion cannot be reached, answer “don’t know". Then, provide an explanation for your choice.:\n{case_text}"""

PROMPT_MC4 = """다음은 가산세 부과에 대한 4지선다 문제입니다. 해당 사례와 각 선택지를 읽고 A, B, C, D 중 가산세 부과에 대해 가장 옳은 선택지 하나만 선택하여 반드시 '답변: A' 와 같이 답하고 설명해주세요. 사례: {case_text}\nA. {a}\nB. {b}\nC. {c}\nD. {d}\n### 답변: """

PROMPT_MC4_ENG = """The following is a multiple-choice question on the imposition of additional tax. Read the case and each option carefully, then choose only one option (A, B, C, or D) that is most correct regarding the imposition of the additional tax. You must respond in the format "Answer: A" and provide an explanation. Case: {case_text}\nA. {a}\nB. {b}\nC. {c}\nD. {d}\n### Answer: """

PROMPT_ESSAY = """당신은 세법의 전문가이며, 법적 문제를 체계적이고 시험 스타일로 다루어야 합니다.

특별히 언급이 없는 한, 한국 세법이 적용된다고 가정하십시오. 다만, 맥락상 정당하다면 한국 세법을 넘어선 법적 쟁점도 다루어야 합니다.

- 정확한 법률 용어를 사용하고, 답변할 때는 반드시 존칭을 사용하십시오.
- 면책조항이나 외부 법률 자문 필요성을 언급하지 마십시오.
- 사용자가 스스로 법을 찾아보도록 요청하지 마십시오.
- 집중된 법적 분석과 개별화된 조언을 제공하십시오.
- 답변은 단호하고 권위 있게 하되, 단순히 일반 정보임을 언급하지 마십시오.
- 한국 특유의 법률 용어를 반드시 포함하십시오.

만약 관련 법적 고려사항을 발견했다면, 간결하고 명확한 법적 분석을 제시하십시오.

항상 구체적인 조문을 인용해야 하며, 조, 항, 호, 목 등을 명시해야 합니다.

예: "국세기본법 제6조 제1항".

- 일반적인 참조(예: "국세기본법")만을 사용하는 것은 허용되지 않습니다.
- 만약 관련성이 있는 정보가 없을 경우, 명시적으로 해당 사항이 없음을 진술해야 합니다.
- 신뢰할 만한 자료가 있으면, 실질적인 지침 또는 통찰을 공유하십시오.
- 답변은 반드시 질문의 언어와 동일한 언어로 하십시오.
- 질문이 "간단한 답변"을 요구하는 경우, 반드시 간결한 답변을 제시하십시오.

아래는 세무사 시험에서 제시된 구체적인 사례를 분석하도록 요구하는 질문인데, 해당 사례 텍스트나 세부사항이 제시되지 않은 경우에는, 필수적인 사례 자료가 누락되었음을 명시적으로 지적해야 합니다.

{question_prefix}
{case_text}
"""

PROMPT_ESSAY_ENG = """You are an expert in tax law and must address legal issues in a systematic, exam-style manner.

Unless otherwise specified, assume that South Korean tax law applies. However, where contextually justified, you should also address legal issues beyond South Korean tax law.

- Use precise legal terminology, and always use honorifics in your responses.
- Do not mention disclaimers or the need for external legal advice.
- Do not ask the user to look up the law themselves.
- Provide focused legal analysis and individualized advice.
- Your answers should be decisive and authoritative; do not state that it is merely general information.
- You must include Korea-specific legal terminology.

If you identify relevant legal considerations, present a concise and clear legal analysis.

You must always cite specific statutory provisions, specifying Article (조), Paragraph (항), Subparagraph (호), and Item (목).

Example: “Article 6, Paragraph 1 of the Framework Act on National Taxes.”

- Using only general references (e.g., “Framework Act on National Taxes”) is not permitted.
- If there is no relevant information, you must explicitly state that none exists.
- If reliable sources are available, share substantive guidance or insights.
- The response must be in the same language as the question.
- If the question requests a “simple answer,” you must provide a concise answer.

If you are asked to analyze a specific case from the certified tax accountant examination, but the case text or details are not provided, you must explicitly point out that the essential case material is missing.

{question_prefix}
{case_text}
"""

PROMPT_ESSAY_JUDGE = """당신은 한국 세법학 시험 답안지를 공적이고 객관적으로 판단하는 채점자입니다.
당신은 세법의 전문가이며, 법적 문제를 체계적이고 시험 스타일로 다루어야 합니다.
목표: 당신의 임무는 수험생의 답변이 모범답안과 제시된 답안이 얼마나 일치하는지를 판단하는 것입니다. 제시된 모범답안을 참고하여 정확성, 완전성, 법적 논증 측면에서 평가하세요.
상황: 당신은 세법학 시험 수험생의 답변과 모범답안을 받게 됩니다.
출력 형태:
답변을 검토한 후:
설명: 답변이 모범답안과 어떻게 부합하거나 벗어나는지를 채점 기준 충족 여부를 근거로 간단히 설명해주세요.
점수는 답변이 모범답안을 얼마나 충족하는지를 반영해야 하며, 다음 형식을 엄격히 따라주세요:
¨ [[점수]] ¨ , 예: ¨ 종합 점수: [[5]] ¨.
종합 점수: 최종 점수를 0에서 10 사이(1 단위)로 부여해주세요.
- 10 = 10개 기준 완전 충족
- 모범답안에 없는 요소가 답변에 포함된 경우, 그것이 법적으로 정확하고 관련성이 있다는 확신이 없는 한 일반적으로 감점해야 함.
- 모범답안은 완전한 답변에 필요한 모든 정보를 포함하는 것으로 가정해야 함.
- 모범답안에 책이나 학술 논문의 경우 인용이 포함될 수 있으나, 답변은 이를 복제할 필요가 없음. 그러나 법 조문은 반드시 정확히 인용해야 하며, 조, 항, 호, 목 등을 명시해야 함.

참고: {rubric}
채점할 수험생의 답안: {model_answer}
모범 답안: {gt}
"""

PROMPT_ESSAY_JUDGE_ENG = """You are a grader objectively and officially evaluating answers for a Korean Tax Law exam. 
You are an expert in tax law, and you must handle legal issues systematically and in an exam style.
Goal: Your task is to determine how closely the candidate’s answer aligns with the model answer provided. Refer to the model answer to evaluate accuracy, completeness, and legal reasoning.
Context: You will be given both a candidate’s answer and a model answer.
Output Format:
After reviewing the answer:
Explanation: Briefly explain how the candidate’s answer aligns with or deviates from the model answer.
Scoring: The score must reflect how well the candidate’s answer fulfills the requirements of the model answer, using the following strict format:
¨ [[Score]] ¨ , Example: ¨ Comprehensive Score: [[5]] ¨
Comprehensive Score: Assign a final score between 0 and 10 (whole numbers only).
- 10 = fully satisfactory (100\%)
- If the candidate includes material not in the model answer, you must generally deduct points unless you are certain it is legally accurate and relevant.
- Assume the model answer contains all information necessary for a complete response.
- If the model answer cites books or academic articles, candidates do not need to reproduce them. However, statutory provisions must be cited accurately with precise identification of Article, Paragraph, Subparagraph, Item, etc.

Reference: {rubric}
채점할 수험생의 답안: {model_answer}
모범 답안: {gt}
"""

# ============================================================================
# Model Clients
# ============================================================================

class OpenAIClient:
    """OpenAI API client wrapper."""
    
    def __init__(self, model_name: str, base_url: str = None, api_key: str = None):
        self.model_name = model_name
        self.client = openai.OpenAI(
            api_key=api_key or os.getenv('OPENAI_API_KEY'),
            base_url=base_url
        )
    
    def generate(self, prompt: str, system_prompt: str = SYSTEM_PROMPT, 
                 temperature: float = 0.0, max_tokens: int = 8192) -> str:
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                temperature=temperature,
                max_tokens=max_tokens
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"OpenAI Error: {e}")
            return None


def get_client(model_name: str, base_url: str = None, api_key: str = None):
    """Factory function to create appropriate client based on model name."""
    return OpenAIClient(model_name, base_url, api_key)


# ============================================================================
# Evaluation Functions
# ============================================================================

def evaluate_mc(
    model: str = "gpt-4o-2024-11-20",
    lang: str = "kor",
    num_samples: int = None,
    output_dir: str = "./eval_results/mc",
    base_url: str = None,
    api_key: str = None
):
    """
    Evaluate on MC (Binary/Ternary) task - Lawful/Unlawful/Unknown classification.
    
    Args:
        model: Model name (supports OpenAI, Claude, Gemini)
        lang: 'kor' or 'eng'
        num_samples: Number of samples to evaluate (None for all)
        output_dir: Directory to save results
        base_url: Custom API base URL (for vLLM etc.)
        api_key: Custom API key
    
    Returns:
        dict: Evaluation results with accuracy and counts
    """
    print(f"\n{'='*60}")
    print(f"Evaluating MC (3-Choice) - {lang.upper()}")
    print(f"Model: {model}")
    print(f"{'='*60}\n")
    
    # Initialize client
    client = get_client(model, base_url, api_key)
    
    # Load dataset
    dataset_name = f"sma1-rmarud/plat-{lang}-mc"
    ds = load_dataset(dataset_name)
    test_data = ds['test']
    
    if num_samples:
        test_data = test_data.select(range(min(num_samples, len(test_data))))
    
    correct = 0
    incorrect = 0
    dontknow = 0
    results = []
    
    prompt_template = PROMPT_MC if lang == "kor" else PROMPT_MC_ENG
    system = SYSTEM_PROMPT if lang == "kor" else SYSTEM_PROMPT_ENG
    
    for idx, data in enumerate(tqdm(test_data, desc="MC Evaluation")):
        # Prepare case text
        case_text = "\n".join([
            data.get('case_info', ''),
            data.get('facts', ''),
            data.get('claims', '')
        ])
        
        prompt = prompt_template.format(case_text=case_text)
        
        # Get model response
        response = client.generate(prompt, system_prompt=system)
        
        if not response:
            dontknow += 1
            results.append({
                "index": idx + 1,
                "prompt": prompt,
                "model_answer": "null",
                "parsed_label": None,
                "result": 2  # dontknow
            })
            continue
        
        # Parse response
        label = extract_first_legality(response)
        
        if label is None:
            dontknow += 1
            result_code = 2
        elif "적법하지" in label:
            result_code = 0  # unlawful
        elif "적법" in label:
            result_code = 1  # lawful
        else:
            result_code = 2  # dontknow
        
        # Ground truth: first 50 are unlawful (0), rest are lawful (1)
        gt = 0 if idx < 50 else 1
        
        if result_code == 2:
            dontknow += 1
        elif result_code == gt:
            correct += 1
        else:
            incorrect += 1
        
        results.append({
            "index": idx + 1,
            "prompt": prompt,
            "model_answer": response,
            "parsed_label": label,
            "result": result_code,
            "gt": gt
        })
        
        if (idx + 1) % 10 == 0:
            total = correct + incorrect
            acc = correct / total * 100 if total > 0 else 0
            print(f"Progress: {idx + 1}/{len(test_data)} | Acc: {acc:.1f}%")
    
    # Calculate final metrics
    total = correct + incorrect
    accuracy = correct / total * 100 if total > 0 else 0
    
    # Save results
    os.makedirs(output_dir, exist_ok=True)
    safe_model_name = model.replace("/", "__")
    
    # Save JSON
    json_path = os.path.join(output_dir, f"{safe_model_name}_{lang}.json")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
    
    # Save CSV
    csv_path = os.path.join(output_dir, f"{safe_model_name}_{lang}_results.csv")
    df = pd.DataFrame([{
        "model": model,
        "lang": lang,
        "correct": correct,
        "incorrect": incorrect,
        "dontknow": dontknow,
        "accuracy": accuracy
    }])
    df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    
    print(f"\n{'='*60}")
    print(f"Results - MC (3-Choice) - {lang.upper()}")
    print(f"{'='*60}")
    print(f"Correct: {correct}")
    print(f"Incorrect: {incorrect}")
    print(f"Don't know: {dontknow}")
    print(f"Accuracy: {accuracy:.2f}%")
    print(f"Results saved to: {output_dir}")
    
    return {
        "correct": correct, 
        "incorrect": incorrect, 
        "dontknow": dontknow, 
        "accuracy": accuracy,
        "results": results
    }


def evaluate_mc4(
    model: str = "gpt-4o-2024-11-20",
    lang: str = "kor",
    num_samples: int = None,
    output_dir: str = "./eval_results/mc4",
    base_url: str = None,
    api_key: str = None
):
    """
    Evaluate on MC4 (4-Choice) task.
    
    Args:
        model: Model name (supports OpenAI, Claude, Gemini)
        lang: 'kor' or 'eng'
        num_samples: Number of samples to evaluate (None for all)
        output_dir: Directory to save results
        base_url: Custom API base URL (for vLLM etc.)
        api_key: Custom API key
    
    Returns:
        dict: Evaluation results with accuracy and counts
    """
    print(f"\n{'='*60}")
    print(f"Evaluating MC4 (4-Choice) - {lang.upper()}")
    print(f"Model: {model}")
    print(f"{'='*60}\n")
    
    # Initialize client
    client = get_client(model, base_url, api_key)
    
    # Load dataset
    dataset_name = f"sma1-rmarud/plat-{lang}-mc4"
    ds = load_dataset(dataset_name)
    test_data = ds['test']
    
    if num_samples:
        test_data = test_data.select(range(min(num_samples, len(test_data))))
    
    correct = 0
    incorrect = 0
    results = []
    
    prompt_template = PROMPT_MC4 if lang == "kor" else PROMPT_MC4_ENG
    system = SYSTEM_PROMPT if lang == "kor" else SYSTEM_PROMPT_ENG
    
    for idx, data in enumerate(tqdm(test_data, desc="MC4 Evaluation")):
        # Prepare case text and choices
        case_text = "\n".join([
            data.get('case_info', ''),
            data.get('facts', ''),
            data.get('claims', '')
        ])
        
        choices = data.get('choices', {})
        a = choices.get('A', '')
        b = choices.get('B', '')
        c = choices.get('C', '')
        d = choices.get('D', '')
        gt = data.get('gt', '')
        
        prompt = prompt_template.format(case_text=case_text, a=a, b=b, c=c, d=d)
        
        # Get model response
        response = client.generate(prompt, system_prompt=system)
        
        if not response:
            results.append({
                "index": idx + 1,
                "prompt": prompt,
                "model_answer": "null",
                "label": None,
                "gt": gt
            })
            continue
        
        # Parse response
        label = extract_first_choice(response)
        
        if label is None:
            incorrect += 1
            print(f"Parsing failed for index {idx + 1}: {response[:100]}...")
        elif label == gt:
            correct += 1
        else:
            incorrect += 1
        
        results.append({
            "index": idx + 1,
            "prompt": prompt,
            "model_answer": response,
            "label": label,
            "gt": gt
        })
        
        if (idx + 1) % 10 == 0:
            total = correct + incorrect
            acc = correct / total * 100 if total > 0 else 0
            print(f"Progress: {idx + 1}/{len(test_data)} | Acc: {acc:.1f}%")
    
    # Calculate final metrics
    total = correct + incorrect
    accuracy = correct / total * 100 if total > 0 else 0
    
    # Save results
    os.makedirs(output_dir, exist_ok=True)
    safe_model_name = model.replace("/", "__")
    
    # Save JSON
    json_path = os.path.join(output_dir, f"{safe_model_name}_{lang}.json")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
    
    # Save CSV
    csv_path = os.path.join(output_dir, f"{safe_model_name}_{lang}_results.csv")
    df = pd.DataFrame([{
        "model": model,
        "lang": lang,
        "correct": correct,
        "incorrect": incorrect,
        "accuracy": accuracy
    }])
    df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    
    # Save per-item results
    result_df = pd.DataFrame([{"parsed_label": r["label"]} for r in results])
    result_csv_path = os.path.join(output_dir, f"{safe_model_name}_{lang}_items.csv")
    result_df.to_csv(result_csv_path, index=False, encoding='utf-8-sig')
    
    print(f"\n{'='*60}")
    print(f"Results - MC4 (4-Choice) - {lang.upper()}")
    print(f"{'='*60}")
    print(f"Correct: {correct}")
    print(f"Incorrect: {incorrect}")
    print(f"Accuracy: {accuracy:.2f}%")
    print(f"Results saved to: {output_dir}")
    
    return {
        "correct": correct, 
        "incorrect": incorrect, 
        "accuracy": accuracy,
        "results": results
    }


def evaluate_essay(
    model: str = "gpt-4o-2024-11-20",
    judge_model: str = "o3-2025-04-16",
    lang: str = "kor",
    num_samples: int = None,
    output_dir: str = "./eval_results/essay",
    base_url: str = None,
    api_key: str = None
):
    """
    Evaluate on Essay task with LLM-as-Judge scoring.
    
    Args:
        model: Model name for generating answers
        judge_model: Model name for judging answers (default: o3)
        lang: 'kor' or 'eng'
        num_samples: Number of samples to evaluate (None for all)
        output_dir: Directory to save results
        base_url: Custom API base URL (for vLLM etc.)
        api_key: Custom API key
    
    Returns:
        dict: Evaluation results with average score
    """
    print(f"\n{'='*60}")
    print(f"Evaluating Essay - {lang.upper()}")
    print(f"Model: {model}")
    print(f"Judge: {judge_model}")
    print(f"{'='*60}\n")
    
    # Initialize clients
    answer_client = get_client(model, base_url, api_key)
    judge_client = get_client(judge_model, base_url, api_key)
    
    # Load dataset
    dataset_name = f"sma1-rmarud/plat-{lang}-essay"
    ds = load_dataset(dataset_name)
    test_data = ds['test']
    
    if num_samples:
        test_data = test_data.select(range(min(num_samples, len(test_data))))
    
    results = []
    points = []

    prompt_template = PROMPT_ESSAY if lang == "kor" else PROMPT_ESSAY_ENG
    system = SYSTEM_PROMPT if lang == "kor" else SYSTEM_PROMPT_ENG
    judge_prompt_template = PROMPT_ESSAY_JUDGE if lang == "kor" else PROMPT_ESSAY_JUDGE_ENG
    
    for idx, data in enumerate(tqdm(test_data, desc="Essay Evaluation")):
        # Prepare question
        question_prefix = data.get('question_prefix', '')
        case_text = "\n".join([
            data.get('case_info', ''),
            data.get('facts', ''),
            data.get('claims', '')
        ])
        prompt = prompt_template.format(question_prefix= question_prefix, case_text=case_text)
        gt = "\n".join([
                data["reasoning"],
                data["decision"]
        ])
        rubric = data.get('rubric', '')
        
        # Step 1: Generate model answer
        response = answer_client.generate(
            prompt, 
            system_prompt=system,
            temperature=0.0
        )
        
        if not response:
            print(f"Failed to generate answer for index {idx + 1}")
            results.append({
                "index": idx + 1,
                "prompt": prompt,
                "model_answer": "null",
                "gt": gt,
                "check": None,
                "point": 0
            })
            points.append(0)
            continue
        
        # Step 2: Judge the answer with rubric
        judge_input = judge_prompt_template.format(rubric=rubric, model_answer=response, gt=gt)
        
        judge_response = judge_client.generate(judge_input, temperature=0.0)
        
        if not judge_response:
            print(f"Failed to judge answer for index {idx + 1}")
            point = 0
        else:
            point = parse_point(judge_response)
        
        points.append(point)
        
        results.append({
            "index": idx + 1,
            "prompt": prompt,
            "model_answer": response,
            "gt": gt,
            "rubric": rubric,
            "check": judge_response,
            "point": point
        })
        
        print(f"Index {idx + 1}: Score = {point}/10")
    
    # Calculate average
    valid_points = [p for p in points if p is not None]
    avg_score = sum(valid_points) / len(valid_points) if valid_points else 0
    
    # Save results
    os.makedirs(output_dir, exist_ok=True)
    safe_model_name = model.replace("/", "__")
    
    # Save JSON
    json_path = os.path.join(output_dir, f"{safe_model_name}_{lang}.json")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
    
    # Save points CSV
    points_df = pd.DataFrame(points + [avg_score], columns=["points"])
    csv_path = os.path.join(output_dir, f"{safe_model_name}_{lang}_points.csv")
    points_df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    
    print(f"\n{'='*60}")
    print(f"Results - Essay - {lang.upper()}")
    print(f"{'='*60}")
    print(f"Average Score: {avg_score:.2f}/10")
    print(f"Results saved to: {output_dir}")
    
    return {
        "average_score": avg_score,
        "points": points,
        "results": results
    }


# ============================================================================
# Main Entry Point
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate LLMs on PLAT dataset")
    parser.add_argument("--task", type=str, choices=["mc", "mc4", "essay", "all"], 
                        default="all", help="Task to evaluate")
    parser.add_argument("--model", type=str, default="gpt-4o-2024-11-20",
                        help="Model name for evaluation")
    parser.add_argument("--judge-model", type=str, default="o3-2025-04-16",
                        help="Model name for essay judging")
    parser.add_argument("--lang", type=str, choices=["kor", "eng", "both"], 
                        default="kor", help="Language")
    parser.add_argument("--num-samples", type=int, default=None,
                        help="Number of samples to evaluate")
    parser.add_argument("--output-dir", type=str, default="./eval_results",
                        help="Output directory for results")
    parser.add_argument("--base-url", type=str, default=None,
                        help="Custom API base URL (for vLLM, etc.)")
    parser.add_argument("--api-key", type=str, default=None,
                        help="Custom API key")
    
    args = parser.parse_args()
    
    languages = ["kor", "eng"] if args.lang == "both" else [args.lang]
    tasks = ["mc", "mc4", "essay"] if args.task == "all" else [args.task]
    
    all_results = {}
    
    for lang in languages:
        for task in tasks:
            key = f"{task}_{lang}"
            
            if task == "mc":
                all_results[key] = evaluate_mc(
                    model=args.model,
                    lang=lang,
                    num_samples=args.num_samples,
                    output_dir=os.path.join(args.output_dir, "mc"),
                    base_url=args.base_url,
                    api_key=args.api_key
                )
            elif task == "mc4":
                all_results[key] = evaluate_mc4(
                    model=args.model,
                    lang=lang,
                    num_samples=args.num_samples,
                    output_dir=os.path.join(args.output_dir, "mc4"),
                    base_url=args.base_url,
                    api_key=args.api_key
                )
            elif task == "essay":
                all_results[key] = evaluate_essay(
                    model=args.model,
                    judge_model=args.judge_model,
                    lang=lang,
                    num_samples=args.num_samples,
                    output_dir=os.path.join(args.output_dir, "essay"),
                    base_url=args.base_url,
                    api_key=args.api_key
                )
    
    # Print summary
    print(f"\n{'='*60}")
    print("EVALUATION SUMMARY")
    print(f"{'='*60}")
    for key, result in all_results.items():
        if "accuracy" in result:
            print(f"{key}: Accuracy = {result['accuracy']:.2f}%")
        elif "average_score" in result:
            print(f"{key}: Average Score = {result['average_score']:.2f}/6")
