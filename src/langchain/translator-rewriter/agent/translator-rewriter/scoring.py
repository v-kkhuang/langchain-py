import json
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


def score_translation(
    source_text: str, translated_text: str, source_lang: str, target_lang: str, style: str = "free"
) -> dict:
    """对翻译结果进行三维度评分。"""

    style_names = {
        "literal": "直译（忠实原文结构）",
        "free": "意译（传达原意，自然流畅）",
        "literary": "文学翻译（信达雅，有文学美感）",
        "business": "商务翻译（专业简洁，正式用语）",
    }

    system_prompt = f"""你是一位严格的翻译质量评审员。
请对以下翻译从三个维度评分（1-10分整数）：
- fluency: 流畅度——译文是否符合{target_lang}的表达习惯
- accuracy: 准确度——译文是否忠实传达了{source_lang}原文的含义
- style_match: 风格匹配度——译文是否符合"{style_names.get(style, style)}"的风格要求

请严格评分，默认从7分开始考虑是否扣分。
以JSON格式输出：
{{"fluency": 8, "accuracy": 9, "style_match": 7, "reason": "简述扣分原因"}}"""

    user_prompt = f"""【原文】({source_lang})
{source_text}

【译文】({target_lang})
{translated_text}

请评分。"""

    response = llm.invoke(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
    )
    try:
        return json.loads(response.content)
    except json.JSONDecodeError:
        return {"fluency": 0, "accuracy": 0, "style_match": 0, "reason": "解析失败"}
