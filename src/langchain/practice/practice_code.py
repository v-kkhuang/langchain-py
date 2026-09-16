import json
import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()
llm = ChatOpenAI(
    model=os.environ["DEEPSEEK_MODEL"],  # 真实模型名
    base_url=os.environ["DEEPSEEK_API_URL"],
    api_key=os.environ["DEEPSEEK_API_KEY"],  # 手动传 key（因为变量名不是 OPENAI_API_KEY）
)




user_prompt =  """
你是一位拥有 15 年经验的高级软件工程师和安全专家。
你精通 Python、JavaScript、Java、Go 多种语言。
你审查代码时会从以下维度全面分析：
1. 功能正确性（Bug）
2. 安全漏洞（SQL 注入、XSS、硬编码密钥等）
3. 性能问题（时间复杂度、内存使用）
4. 可维护性（命名、注释、代码风格）
5. 最佳实践（是否遵循语言惯用法）

对每个问题，给出：严重程度（🔴 严重 / 🟡 警告 / 🔵 建议）+ 修复代码。

请审查以下 Python 代码，指出问题：

def calculate_average(numbers):
    total = 0
    for i in range(len(numbers)):
        total += numbers[i]
    return total / len(numbers)

"""


response = llm.invoke(
     [
            {"role": "user", "content": user_prompt},
        ]
)
print(response.content)

