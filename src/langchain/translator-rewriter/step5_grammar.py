import json
import os
import sys
from typing import Literal

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

sys.stdout.reconfigure(encoding="utf-8")

load_dotenv()


llm = ChatOpenAI(
    model=os.getenv("DEEPSEEK_MODEL"),
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url=os.getenv("DEEPSEEK_API_URL"),
)


def correct_grammar(text: str, lang: str = "中文") -> dict:
    """
    语法纠错：返回修正后的文本和修改说明。
    使用 JSON 输出，方便程序处理。
    """
    system_prompt = f"""你是一位严谨的{lang}校对编辑。
请检查以下文本中的语法和拼写错误，并进行修正。

要求：
1. 只修正明确的语法错误和拼写错误
2. 不改变原文的文体风格和语气
3. 不优化表达，只纠正错误

请以 JSON 格式输出，不要输出其他内容：
{{"corrected": "修正后的文本", "errors": [{{"original": "错误片段", "corrected": "修正片段", "reason": "错误原因"}}], "error_count": 2}}"""

    response = llm.invoke(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": text},
        ]
    )

    try:
        return json.loads(response.content)
    except json.JSONDecodeError:
        return {
            "corrected": response.content,
            "errors": [],
            "error_count": 0,
            "parse_error": True,
        }


# 测试
test_texts = [
    "这个方案的执行过程中遇到了一些困难，我们需要重新评估它。",  # 可能无错误
    "The quick brown fox jump over the lazy dog.",  # jump→jumps
    "最近我們公司的業績有所改善，但是仍需要努力。",  # 可能是繁简混用
]

for text in test_texts:
    result = correct_grammar(text)
    print(f"\n原文: {text}")
    print(f"修正: {result.get('corrected', 'N/A')}")
    print(f"错误数: {result.get('error_count', 0)}")
    for err in result.get("errors", []):
        print(f"  {err['original']} → {err['corrected']} ({err['reason']})")
