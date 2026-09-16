import json
import os
import sys

import step3_style
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

sys.stdout.reconfigure(encoding="utf-8")

load_dotenv()


llm = ChatOpenAI(
    model=os.getenv("DEEPSEEK_MODEL"),
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url=os.getenv("DEEPSEEK_API_URL"),
)


class Glossary:
    """术语表：维护术语的统一翻译映射。"""

    def __init__(self):
        self.terms = {}  # {"源语言术语": "目标语言翻译"}

    def add(self, source: str, target: str):
        self.terms[source] = target

    def load_from_file(self, filepath: str):
        with open(filepath, "r", encoding="utf-8") as f:
            self.terms = json.load(f)

    def format_for_prompt(self) -> str:
        if not self.terms:
            return ""
        lines = ["请确保以下术语翻译一致："]
        for src, tgt in self.terms.items():
            lines.append(f"  {src} → {tgt}")
        return "\n".join(lines)


def translate_with_glossary(
    text: str,
    glossary: Glossary,
    source_lang: str = "中文",
    target_lang: str = "English",
    style: str = "free",
) -> str:
    """带术语表的翻译。"""
    config = step3_style.STYLE_CONFIG[style]

    system_prompt = f"{config['role']}\n源语言：{source_lang}\n目标语言：{target_lang}"

    glossary_text = glossary.format_for_prompt()
    user_prompt = f"{config['instruction']}\n\n"
    if glossary_text:
        user_prompt += f"{glossary_text}\n\n"
    user_prompt += f"请翻译：\n{text}"
    print(system_prompt)
    print("--------------------------------------------------------------")
    print(user_prompt)
    response = llm.invoke(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
    )
    return response.content


# 使用示例
glossary = Glossary()
glossary.add("人工智能", "Artificial Intelligence")
glossary.add("深度学习", "Deep Learning")
glossary.add("大语言模型", "Large Language Model")

# 翻译多段文本，术语保持一致
paragraphs = [
    "人工智能正在快速发展。",
    "深度学习是人工智能的一个分支。",
    "大语言模型是深度学习的重要应用。",
]

for i, p in enumerate(paragraphs):
    result = translate_with_glossary(p, glossary, "中文", "English")
    print(f"段落 {i + 1}: {result}")


# 方式 A：术语表注入

# 方式 B：对话记忆（多轮翻译）


# 对话记忆的 token 成本
# 方式 B 会随着段落数增加，messages 列表越来越长，token 成本线性增长。如果文档超过 20 段，
# 建议切换为方式 A（术语表），或者每 5 段提取一次术语更新术语表，然后清空对话历史重新开始。
# 这是工程上的权衡——   一致性 vs 成本。
