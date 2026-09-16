import os
from langchain_core import prompts
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()
llm = ChatOpenAI(
    model=os.environ["DEEPSEEK_MODEL"],  # 真实模型名
    base_url=os.environ["DEEPSEEK_API_URL"],
    api_key=os.environ["DEEPSEEK_API_KEY"],  # 手动传 key（因为变量名不是 OPENAI_API_KEY）
)

# 简单的系统提示词，直接使用 system

response = llm.invoke(
    [
        {"role": "user", "content": "说一句你好"},
    ]
)

print(response.content)
print(response.additional_kwargs.get("reasoning_content"))
print(response)
