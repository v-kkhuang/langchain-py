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

RewriteMode = Literal["formalize", "casualize", "simplify", "expand"]

REWRITE_CONFIG = {
    "formalize": {
        "role": "你是一位资深文字编辑，擅长将口语化文本改写为正式书面语。",
        "instruction": "请将以下文本改写为正式书面语，保持原意不变。使用规范的书面用语，避免口语、缩略语和网络用语。",
    },
    "casualize": {
        "role": "你是一位擅长日常表达的作家，能把正式文本改写为通俗易懂的口语。",
        "instruction": "请将以下文本改写为日常口语，保持原意不变。使用自然的口语表达，就像和朋友聊天一样。",
    },
    "simplify": {
        "role": "你是一位科普作家，擅长把复杂内容简化为普通人能理解的语言。",
        "instruction": "请简化以下文本，保持核心信息不变，但降低阅读难度。使用短句和常见词汇，避免专业术语。如果原文有术语，请用通俗的方式解释。",
    },
    "expand": {
        "role": "你是一位文笔细腻的作家，擅长在保持原意的基础上扩充细节。",
        "instruction": "请在保持原意的基础上扩写以下文本。补充逻辑过渡、场景描写和细节，但不要编造原文没有的事实信息。",
    },
}


def rewrite(text: str, mode: RewriteMode = "formalize") -> str:
    config = REWRITE_CONFIG[mode]
    user_prompt = f"{config['instruction']}\n\n原文：\n{text}"

    response = llm.invoke(
        [
            {"role": "system", "content": config["role"]},
            {"role": "user", "content": user_prompt},
        ]
    )
    return response.content


# 测试
# sample = "那个新出的AI工具贼好用，搞东西特别快，大家都在用。"
# for m in ["formalize", "simplify", "expand"]:
#     print(f"\n=== {m} ===")
#     print(rewrite(sample, m))

# 测试链
sample = "今天天气气温负8度，大家请注意保暖！"

print("\n=== casualize ===")
print(rewrite(sample, "casualize"))

print("\n=== formalize ===")
print(rewrite(sample, "formalize"))

print("\n=== simplify ===")
print(rewrite(sample, "simplify"))
