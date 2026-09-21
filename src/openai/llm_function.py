import os

from dotenv import load_dotenv

from openai import OpenAI

load_dotenv()

llm = OpenAI(api_key=os.environ["DEEPSEEK_API_KEY"], base_url=os.environ["DEEPSEEK_API_URL"])


def get_weather(city: str) -> str:
    123


tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "查询天气",
            "parameters": {
                "type": "Object",
                "propertites": {"city": {"type": "str", "description": "城市名称"}},
            },
        },
        "required": ["city"],
    }
]


response = llm.chat.completions.create(
    model=os.environ["DEEPSEEK_MODEL"],
    temperature=0.7,
    messages=[{"role": "user", "content": "你好，我叫黄开,请记住我"}],
    stream=True,
    tools=tools,
)
