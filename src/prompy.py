import sys
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate

# Windows 终端默认 GBK，强制 UTF-8 输出避免 emoji 崩溃
sys.stdout.reconfigure(encoding="utf-8")

load_dotenv()  # 从 .env 读取 DEEPSEEK_API_KEY


# 3. 创建PromptTemplate实例
prompt = PromptTemplate(
    input_variables=["product", "feature"],  # 列出所有变量名
    template="你是一名{product}工程师，请帮我设计{feature}的单元测试用例",  # 传入模板字符串
)
formatted_prompt = prompt.format(product="资深测试", feature="条件覆盖")
print(formatted_prompt)


promt2 = PromptTemplate.from_template("你是一名{product}工程师，请帮我设计{feature}的单元测试用例")
formatted_prompt2 = promt2.invoke({"product": "开发", "feature": "冒烟"})
print(formatted_prompt2)

# ChatDeepSeek 自动读取 DEEPSEEK_API_KEY，无需手动传 key 或 base_url
# llm = ChatOpenAI(
#     model="glm-5.1",                          # 真实模型名
#     base_url="",
#     api_key=os.environ["DEEPSEEK_API_KEY"],         # 手动传 key（因为变量名不是 OPENAI_API_KEY）
#     temperature=2
# )


# for chunk in llm.invoke([{"role": "user", "content": "介绍下苹果手机，30字"}]):
#     # 每个 chunk 是一小段 AIMessageChunk
#     collected += chunk.content
#     print(chunk.content, end="", flush=True)
#     print()  # 换行
#     print(f"🔍 共收到 {len(collected)} 个字符")
