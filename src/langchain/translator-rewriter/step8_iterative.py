import json
import os
import sys

import step3_style
import step6_score
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

sys.stdout.reconfigure(encoding="utf-8")

load_dotenv()


llm = ChatOpenAI(
    model=os.getenv("DEEPSEEK_MODEL"),
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url=os.getenv("DEEPSEEK_API_URL"),
)


def translate_iterative(
    text: str,
    source_lang: str = "中文",
    target_lang: str = "English",
    style: str = "free",
    max_iterations: int = 3,
    threshold: float = 8.0,
) -> dict:
    """
    迭代优化翻译：
    1. 翻译
    2. 评分
    3. 如果均分 < threshold，把评分理由加入 prompt 重新翻译
    4. 最多迭代 max_iterations 次
    """
    config = step3_style.STYLE_CONFIG[style]
    system_prompt = f"{config['role']}\n源语言：{source_lang}\n目标语言：{target_lang}"
    base_instruction = config["instruction"]
    feedback = ""

    for i in range(max_iterations):
        user_prompt = base_instruction
        if feedback:
            user_prompt += f"\n\n【上一轮翻译的问题，请改进】\n{feedback}"
        user_prompt += f"\n\n请翻译：\n{text}"

        response = llm.invoke(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]
        )
        translated = response.content

        scores = step6_score.score_translation(text, translated, source_lang, target_lang, style)
        avg = (scores["fluency"] + scores["accuracy"] + scores["style_match"]) / 3

        print(f"  第 {i + 1} 轮 | 均分: {avg:.1f} | {scores['reason']}")

        if avg >= threshold:
            return {"translation": translated, "scores": scores, "iterations": i + 1}

        feedback = scores["reason"]

    return {"translation": translated, "scores": scores, "iterations": max_iterations}


# 测试
result = translate_iterative(
    "这项技术的突破为整个行业带来了前所未有的发展机遇", "中文", "English", "business"
)
print(f"\n最终译文（{result['iterations']}轮）: {result['translation']}")
print(f"最终评分: {result['scores']}")
