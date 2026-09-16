import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from typing import Literal


load_dotenv()


llm = ChatOpenAI(
    model=os.environ["DEEPSEEK_MODEL"],
    api_key=os.environ["DEEPSEEK_API_KEY"],
    base_url=os.environ["DEEPSEEK_API_URL"],
)

# 摘要质量评分
# 让模型给自己的摘要打分。这不是"锦上添花"——自评分数是下一步迭代优化的前提条件：没有度量就没有改进方向。
# 三个评分维度
# 维度	含义	1 分意味着	10 分意味着
# 覆盖度	摘要是否涵盖了原文的关键信息	遗漏了主要观点	所有关键信息都在
# 简洁度	摘要是否去除了冗余信息	几乎和原文一样长	极精炼，无废话
# 准确度	摘要内容是否忠实于原文，无捏造	有事实错误或编造	完全忠实原文

# 实现方式：二次调用
# 评分思路：
# 先调用 API 生成摘要，
# 再第二次调用 API，把原文和摘要一起给它，让它评分。
# 为什么不在一次调用中同时生成摘要和评分？
# 因为分开调用可以避免模型"为了拿高分而给自己放水"——先评后改是更可靠的流程。


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


text = """
人工智能（AI）是计算机科学的一个分支，致力于创造能够模拟人类智能行为的系统。
近年来，深度学习技术的突破使得AI在图像识别、自然语言处理、语音识别等领域取得了显著进展。
然而，AI的发展也带来了伦理挑战，包括隐私保护、算法偏见、就业影响等问题。
专家认为，未来AI的发展需要在技术创新和伦理约束之间找到平衡。
"""


# 简洁模式 + 正式语气 + 中文
summary_text = summarize(text, mode="concise", tone="formal", lang="zh")
response = score_summary(text, summary_text)
print(response.content)

# 要点模式 + 通俗语气 + 英文
summary_text2 = summarize(text, mode="bullet", tone="casual", lang="en")
response = score_summary(text, summary_text2)
print(response.content)

# 详细模式 + 学术语气 + 中文
summary_text3 = summarize(text, mode="detailed", tone="academic", lang="zh")
response = score_summary(text, summary_text3)
print(response.content)

# 随意模式
summary_text4 = "AI能够模拟人类行为,专家还没有平衡创新和伦理"
response = score_summary(text, summary_text4)
print(response.content)
