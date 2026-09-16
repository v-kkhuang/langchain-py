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

Style = Literal["literal", "free", "literary", "business"]

STYLE_CONFIG = {
    "literal": {
        "role": "你是一位严谨的翻译家，风格为直译。\
        始终忠实于原文的句子结构，逐句对应翻译，不增加或遗漏信息。",
        "instruction": "请保持原文的句序和结构，不要意译或改写。",
    },
    "free": {
        "role": "你是一位经验丰富的翻译家，风格为意译。\
        以传达原文含义为首要目标，译文应符合目标语言的表达习惯。",
        "instruction": "不必拘泥于原文句式，可以调整语序以使译文更自然。",
    },
    "literary": {
        "role": "你是一位文学翻译家，追求信达雅。译文应具有文学美感和可读性，用词优美，行文流畅。",
        "instruction": "注重译文的文学性，可以适当使用修辞手法。",
    },
    "business": {
        "role": "你是一位商务翻译专家，擅长翻译商业文档。用词专业简洁，符合商务文书规范。",
        "instruction": "使用正式商务用语，避免口语化表达。",
    },
}


def translate(
    text: str,
    source_lang: str = "中文",
    target_lang: str = "English",
    style: Style = "free",
) -> str:
    config = STYLE_CONFIG[style]

    system_prompt = f"{config['role']}\n源语言：{source_lang}\n目标语言：{target_lang}"
    user_prompt = f"{config['instruction']}\n\n请翻译：\n{text}"

    response = llm.invoke(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
    )
    return response.content


# 测试：四种风格翻译同一段文本
if __name__ == "__main__":
    sample = "这家初创公司凭借其颠覆性的技术，在短短两年内就颠覆了整个行业格局。"

    for s in ["literal", "free", "literary", "business"]:
        print(f"\n=== {s} ===")
        print(translate(sample, "中文", "English", s))
