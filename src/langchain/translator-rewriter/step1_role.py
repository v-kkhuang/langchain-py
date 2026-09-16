import os
import sys

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

sys.stdout.reconfigure(encoding="utf-8")

load_dotenv()


llm = ChatOpenAI(
    model=os.getenv("DEEPSEEK_MODEL"),
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url=os.getenv("DEEPSEEK_API_URL"),
)


def translate_with_role(
    text: str, source_lang: str = "中文", target_lang: str = "English", role: str = "translator"
) -> str:
    """带角色提示的翻译。"""
    role_map = {
        "translator": "你是一位精通多语言的专业翻译家，追求准确传达原文含义。",
        "literary": "你是一位文学翻译家，注重译文的文学性和可读性，追求信达雅。",
        "business": "你是一位商务翻译专家，擅长翻译商业文档，用词专业简洁。",
        "editor": "你是一位资深文字编辑，擅长在保持原意的基础上优化文字表达。",
    }

    system_prompt = role_map.get(role, role_map["translator"])
    user_prompt = f"请将以下{source_lang}翻译为{target_lang}：\n\n{text}"
    response = llm.invoke(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
    )
    return response.content


# 对比不同角色的翻译
sample = "今天真倒霉，我把玩具弄丢了！"

for role in ["translator", "literary", "business", "editor"]:
    print(f"\n=== 角色: {role} ===")
    print(translate_with_role(sample, "中文", "English", role))


# === 角色: translator ===
# What a terrible day! I lost my toy!

# === 角色: literary ===
# What a bad day! I lost my toy!

# === 角色: business ===
# What awful luck today! I lost my toy!

# === 角色: editor ===
# What a terrible day—I lost my toy!
