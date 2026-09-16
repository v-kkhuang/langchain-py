import json
import os
from typing import Literal

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()


llm = ChatOpenAI(
    model=os.environ["DEEPSEEK_MODEL"],
    api_key=os.environ["DEEPSEEK_API_KEY"],
    base_url=os.environ["DEEPSEEK_API_URL"],
)


# 迭代提示优化
# 迭代流程：
# 摘要->评分 -> 如果低于阈值 ，分析问题 ->修正prompt ->重新生成
Mode = Literal["concise", "bullet", "detailed"]
Tone = Literal["formal", "casual", "academic"]
Lang = Literal["zh", "en"]


def build_prompt(
    text: str, mode: Mode = "concise", tone: Tone = "formal", lang: Lang = "zh"
) -> tuple[str, str]:
    """构建 system prompt 和 user prompt。"""
    # 角色描述（融入语气）
    tone_map = {
        "formal": "使用正式、专业的语气",
        "casual": "使用通俗易懂、口语化的语气",
        "academic": "使用学术写作风格，用词严谨，逻辑清晰",
    }
    lang_map = {"zh": "中文", "en": "English"}

    system_prompt = f"你是一位专业的文本摘要专家。{tone_map[tone]}。输出语言：{lang_map[lang]}。"

    # 模式对应的格式约束
    mode_instructions = {
        "concise": "请用不超过100字的一段话概括文本核心内容。不要使用列表。",
        "bullet": "请提取3到5个要点，用列表格式输出。每个要点一行，以 - 开头。",
        "detailed": "请输出结构化摘要，包含以下部分：\n1. 核心主旨（1-2句）\n2. 关键论点（逐条列出）\n3. 结论与展望（1-2句）",
    }

    user_prompt = f"""{mode_instructions[mode]}

请总结以下文本：

{text}"""

    return system_prompt, user_prompt


def summarize(text: str, mode: Mode = "concise", tone: Tone = "formal", lang: Lang = "zh") -> str:
    system_prompt, user_prompt = build_prompt(text, mode, tone, lang)
    response = llm.invoke(
        [
            {"role": "user", "content": user_prompt},
            {"role": "system", "content": system_prompt},
        ]
    )
    return response.content


# 原文和概括都传给大模型llm
def score_summary(original_text: str, summary: str) -> str:
    """对摘要进行质量评分，返回三个维度的分数和理由。"""
    system_prompt = """你是一位严格的摘要质量评审员。
你需要对给定的摘要从三个维度评分（1-10分整数）：
- coverage: 覆盖度——摘要是否涵盖了原文的关键信息
- conciseness: 简洁度——摘要是否精炼无冗余
- accuracy: 准确度——摘要是否忠实于原文，无捏造

请以 JSON 格式输出，不要输出任何其他内容：
{"coverage": 8, "conciseness": 7, "accuracy": 9, "reason": "简短说明扣分原因"}

请严格评分，默认从 7 分开始考虑是否扣分
"""

    user_prompt = f"""【原文】
{original_text}

【摘要】
{summary}

请评分。"""
    return llm.invoke(
        [
            {"role": "user", "content": user_prompt},
            {"role": "system", "content": system_prompt},
        ]
    )


sample_text = """
人工智能（AI）是计算机科学的一个分支，致力于创造能够模拟人类智能行为的系统。
近年来，深度学习技术的突破使得AI在图像识别、自然语言处理、语音识别等领域取得了显著进展。
然而，AI的发展也带来了伦理挑战，包括隐私保护、算法偏见、就业影响等问题。
专家认为，未来AI的发展需要在技术创新和伦理约束之间找到平衡。
"""


def summarize_iterative(
    text: str,
    mode: str = "concise",
    tone: str = "formal",
    lang: str = "zh",
    max_iterations: int = 3,
    threshold: float = 8.0,
) -> dict:
    """
    迭代优化摘要：
    1. 生成摘要
    2. 评分
    3. 如果均分 < threshold，把评分理由加入 prompt 重新生成
    4. 最多迭代 max_iterations 次
    """
    system_prompt, user_prompt = build_prompt(text, mode, tone, lang)
    feedback = ""  # 上一轮的反馈信息

    for i in range(max_iterations):
        # 如果有反馈，追加到 user prompt
        current_user_prompt = user_prompt
        current_system_prompt = system_prompt
        if feedback:
            current_user_prompt += f"\n\n【上一轮摘要的问题，请改进】\n{feedback}"
        summary = llm.invoke(
            [
                {"role": "user", "content": current_user_prompt},
                {"role": "system", "content": current_system_prompt},
            ]
        )
        # 评分
        print(summary.content)
        scores = json.loads(score_summary(text, summary.content).content)
        print(scores)
        avg_score = (scores["coverage"] + scores["conciseness"] + scores["accuracy"]) / 3
        print(f"  第 {i + 1} 轮: 均分 {avg_score:.1f} | {scores['reason']}")
        # 达标则停止
        if avg_score >= threshold:
            return {"summary": summary.content, "scores": scores, "iterations": i + 1}
        # 未达标：把评分理由作为下一轮的反馈
        feedback = scores["reason"]
    return {"summary": summary.content, "scores": scores, "iterations": max_iterations}


# 测试
result = summarize_iterative(sample_text)
print(f"\n最终摘要（{result['iterations']}轮）: {result['summary']}")
print(f"最终评分: {result['scores']}")
