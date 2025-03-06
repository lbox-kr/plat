from datasets import load_dataset
import os
from dotenv import load_dotenv
import re
import openai

load_dotenv()

client = openai.OpenAI(
    api_key=os.getenv('OPENAI_API_KEY')
)

def gpt_decision(processed_precedent):
    try:
        response = client.chat.completions.create(
            model="gpt-4o-2024-08-06",
            messages=[
                {"role": "system", "content": "당신은 사용자의 질문에 친절하고 논리적으로 답변해 주는 세무 전문가 챗봇입니다."},
                {"role": "user", "content": f"위 제시된 가산세 부과처분의 배경을 읽고 해당 가산세 부과에 대해 적법한지 적법하지 않은지, \"적법함\", \"적법하지 않음\" 둘 중 하나를 답변하고 설명해주세요. 답을 내릴 수 없다면 \"알수없음\"으로 답변하고 설명해주세요.: {processed_precedent} ###답변 : "}
            ],
            temperature=0.3
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error occurred: {e}")
        return None

if __name__ == "__main__":
    correct = 0
    wrong = 0
    dontknow = 0

    ds = load_dataset("sma1-rmarud/PLAT", "mc")

    for idx in range(3):  
        data = ds['test'][idx]

        processed_precedent = "\n".join([
            data.get("case_info", ""),  
            data.get("facts", ""),  
            data.get("claims", "")  
        ])

        gpt_decision_ans = gpt_decision(processed_precedent)

        if not gpt_decision_ans:
            print(f"GPT has no response (Index {idx})")
            continue  

        lawful_pattern = r"^적법함$"
        unlawful_pattern = r"^적법하지\s?않음$"
        dontknow_pattern = r"^알\s?수\s?없음$"

        if re.fullmatch(dontknow_pattern, gpt_decision_ans):
            dontknow += 1
        elif idx < 25:
            if re.fullmatch(unlawful_pattern, gpt_decision_ans):
                correct += 1
            else:
                wrong += 1
        else:
            if re.fullmatch(lawful_pattern, gpt_decision_ans):
                correct += 1
            else:
                wrong += 1

    # 최종 결과 출력
    print(f"✅ correct num count: {correct}")
    print(f"❌ wrong num count: {wrong}")
    print(f"🤔 dontknow num count: {dontknow}")
