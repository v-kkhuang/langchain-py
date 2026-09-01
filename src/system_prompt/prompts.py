import os
from langchain_core import prompts
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv


# langchain 框架中message 分为四种
# SystemMessage：系统提示词，prompts 常用位置
# HumanMessage:用户输入问题
# AIMessage:  大模型返回
# ToolMessage: 工具返回

load_dotenv()
llm = ChatOpenAI(
    model=os.environ["DEEPSEEK_MODEL"],  # 真实模型名
    base_url=os.environ["DEEPSEEK_API_URL"],
    api_key=os.environ["DEEPSEEK_API_KEY"],  # 手动传 key（因为变量名不是 OPENAI_API_KEY）
    extra_body={"enable_thinking": False},
)

# 简单的系统提示词，直接使用 system

response = llm.invoke(
    [
        {"role": "system", "content": "你是一个智能问答助手，回答用户的问题"},
        {"role": "user", "content": "1加3乘以3=多少"},
    ]
)

print(response.content)
print(response.additional_kwargs.get("reasoning_content"))
print(response)
