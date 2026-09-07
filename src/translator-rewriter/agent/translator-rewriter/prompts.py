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
RewriteMode = Literal["formalize", "casualize", "simplify", "expand"]

STYLE_CONFIG = {
    "literal": {
        "role": "你是一位严谨的翻译家，风格为直译。始终忠实于原文的句子结构，逐句对应翻译，不增加或遗漏信息。",
        "instruction": "请保持原文的句序和结构，不要意译或改写。",
    },
    "free": {
        "role": "你是一位经验丰富的翻译家，风格为意译。以传达原文含义为首要目标，译文应符合目标语言的表达习惯。",
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

# Few-shot 示例库——按领域和风格组织
FEW_SHOT_EXAMPLES = {
    "tech_literal": [
        {
            "source": "人工智能正在改变我们的生活方式",
            "target": "Artificial intelligence is changing our way of life",
        },
        {
            "source": "深度学习模型需要大量训练数据",
            "target": "Deep learning models require large amounts of training data",
        },
    ],
    "literary_free": [
        {
            "source": "夕阳西下，金色的余晖洒满了整座城市",
            "target": "As the sun dipped below the horizon, \
             golden afterglow bathed the entire city in warmth",
        },
        {"source": "她的笑容像春风一样温暖", "target": "Her smile was as warm as a spring breeze"},
    ],
    "business_formal": [
        {
            "source": "我们对本季度的业绩表现感到满意",
            "target": "We are pleased with our performance for this quarter",
        },
        {
            "source": "请贵方尽快确认合同条款",
            "target": "We kindly request your prompt confirmation of the contract terms",
        },
    ],
}


def build_fewshot_prompt(
    text: str, source_lang: str, target_lang: str, example_key: str = "tech_literal"
) -> list:
    """构建带 Few-shot 示例的 messages 列表。"""
    examples = FEW_SHOT_EXAMPLES.get(example_key, [])

    system_prompt = (
        f"你是一位专业翻译家。请将{source_lang}翻译为{target_lang}。参考以下示例的风格和术语选择。"
    )

    messages = [{"role": "system", "content": system_prompt}]

    # 每个示例作为一组 user/assistant 对话
    for ex in examples:
        messages.append({"role": "user", "content": ex["source"]})
        messages.append({"role": "assistant", "content": ex["target"]})

    # 最后是真正要翻译的文本
    messages.append({"role": "user", "content": text})

    return messages


def translate_fewshot(
    text: str,
    source_lang: str = "中文",
    target_lang: str = "English",
    example_key: str = "tech_literal",
) -> str:
    messages = build_fewshot_prompt(text, source_lang, target_lang, example_key)
    response = llm.invoke(messages)
    return response.content
