import os
from typing import Literal

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()
llm = ChatOpenAI(
    model=os.environ["DEEPSEEK_MODEL"],  # 真实模型名
    base_url=os.environ["DEEPSEEK_API_URL"],
    api_key=os.environ["DEEPSEEK_API_KEY"],  # 手动传 key（因为变量名不是 OPENAI_API_KEY）
)


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


text = """
人工智能（AI）是计算机科学的一个分支，致力于创造能够模拟人类智能行为的系统。
近年来，深度学习技术的突破使得AI在图像识别、自然语言处理、语音识别等领域取得了显著进展。
然而，AI的发展也带来了伦理挑战，包括隐私保护、算法偏见、就业影响等问题。
专家认为，未来AI的发展需要在技术创新和伦理约束之间找到平衡。
"""


# 简洁模式 + 正式语气 + 中文
print(summarize(text, mode="concise", tone="formal", lang="zh"))

# 要点模式 + 通俗语气 + 英文
print(summarize(text, mode="bullet", tone="casual", lang="en"))

# 详细模式 + 学术语气 + 中文
print(summarize(text, mode="detailed", tone="academic", lang="zh"))


# 人工智能是模拟人类智能的计算机科学，深度学习推动了图像识别、自然语言处理等领域的突破，但同时也带来隐私保护、算法偏见和就业影响等伦理挑战。未来AI发展需在技术创新与伦理约束之间寻求平衡。


# - AI is a branch of computer science focused on creating systems that mimic human intelligence.
# - Recent breakthroughs in deep learning have driven major advances in image recognition, natural language processing, and speech recognition.
# - AI development raises ethical challenges, including privacy protection, algorithmic bias, and impacts on employment.
# - Experts believe the future of AI requires finding a balance between technological innovation and ethical constraints.


# ### 核心主旨

# 人工智能作为计算机科学的重要分支，通过深度学习等技术在众多领域取得显著突破，但其健康发展亟需在技术创新与伦理约束之间实现动态平衡。

# ### 关键论点

# - **定义与范畴**：人工智能旨在模拟人类智能行为，属于计算机科学的核心研究领域。
# - **技术突破**：深度学习的进步显著推动了图像识别、自然语言处理及语音识别等关键应用的发展。
# - **伦理挑战**：AI的快速普及引发了隐私保护不足、算法歧视风险以及就业结构冲击等现实问题。
# - **未来取向**：专家强调，寻求技术创新与伦理规范的协同共进是AI持续发展的必要条件。

# ### 结论与展望

# 未来人工智能的发展将不仅是技术能力的竞争，更是治理智慧与伦理框架的考验。若能在创新动能与人文约束间建立有效机制，AI有望在提升社会福祉的同时降低系统性风险。


# 总结：提示词模板适用于，同一个应用场景下，不同的角色或者不同的处理办法时适用
