import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from typing import Literal

load_dotenv()


llm = ChatOpenAI(
    model=os.getenv["DEEPSEEK_MODEL"],
    api_key=os.getenv["DEEPSEEK_API_KEY"],
    base_url=os.getenv["DEEPSEEK_API_URL"],
)


text = """
人工智能（AI）是计算机科学的一个分支，致力于创造能够模拟人类智能行为的系统。
近年来，深度学习技术的突破使得AI在图像识别、自然语言处理、语音识别等领域取得了显著进展。
然而，AI的发展也带来了伦理挑战，包括隐私保护、算法偏见、就业影响等问题。
专家认为，未来AI的发展需要在技术创新和伦理约束之间找到平衡。
"""
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
                    {text}
                    """
    return system_prompt, user_prompt


llm.invoke()
