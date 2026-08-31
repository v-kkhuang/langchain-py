import sys
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate

# Windows 终端默认 GBK，强制 UTF-8 输出避免 emoji 崩溃
sys.stdout.reconfigure(encoding="utf-8")

load_dotenv()  # 从 .env 读取 DEEPSEEK_API_KEY

# ChatDeepSeek 自动读取 DEEPSEEK_API_KEY，无需手动传 key 或 base_url
llm = ChatOpenAI(
    model="glm-5.1",                          # 真实模型名
    base_url="http://172.21.3.106",
    api_key=os.environ["DEEPSEEK_API_KEY"],         # 手动传 key（因为变量名不是 OPENAI_API_KEY）
    temperature=2
)


tpl = PromptTemplate.from_template("解释{topic}，受众是{audience}。")
tpl.input_variables  # ['topic', 'audience']
tpl.format(topic="hi",audience="me")
chain = tpl|llm
result = chain.invoke({"topic":"hi","audience":"me"})
print(result.content)


# for chunk in llm.invoke([{"role": "user", "content": "介绍下苹果手机，30字"}]):
#     # 每个 chunk 是一小段 AIMessageChunk
#     collected += chunk.content
#     print(chunk.content, end="", flush=True)
#     print()  # 换行
#     print(f"🔍 共收到 {len(collected)} 个字符")