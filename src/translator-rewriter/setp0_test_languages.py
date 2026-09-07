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

user_prompt = "请用{lang_name}回复以下句子的翻译：你好，世界"


def test_language(lang_name: str):

    response = llm.invoke(
        [
            {"role": "user", "content": user_prompt.format(lang_name=lang_name)},
        ]
    )
    print(response.content)


for lang in ["英语", "日语", "韩语", "法语", "西班牙语"]:
    test_language(lang)
