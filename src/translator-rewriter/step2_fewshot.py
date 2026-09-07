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
            "target": "As the sun dipped below the horizon, golden afterglow bathed the entire city in warmth",
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


# 测试：同一文本，不同 Few-shot 风格
sample = "这项技术突破为行业带来了前所未有的机遇"

for key in ["tech_literal", "business_formal", "literary_free"]:
    print(f"\n=== 风格: {key} ===")
    print(translate_fewshot(sample, "中文", "English", key))
