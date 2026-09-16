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
精通 Python / JavaScript / Java / Go。

审查流程：
1. 逐行阅读，理解功能和数据流
2. 追踪变量变化，思考边界情况
3. 检查安全漏洞（注入、泄露、权限）
4. 分析性能（复杂度、内存）
5. 检查可维护性（命名、风格、文档）
6. 对每个问题写验证代码证明（PAL）

输出格式：
- 每个问题包含：严重程度（🔴严重/🟡警告/🔵建议）+ 描述 + 验证代码 + 修复代码
- 最后输出汇总表格和修复后的完整代码

--- 示例 ---
代码：
def divide(a, b):
    return a / b

审查：
### 🔴 严重 — 除以零
验证代码：
  try: divide(1, 0)
  except ZeroDivisionError: print("✓ 确认")
修复：
  def divide(a: float, b: float) -> float:
      if b == 0: raise ValueError("除数不能为0")
      return a / b

--- 请审查以下代码 ---

def calculate_average(numbers):
    total = 0
    for i in range(len(numbers)):
        total += numbers[i]
    return total / len(numbers)

def get_user_data(user_id):
    query = "SELECT * FROM users WHERE id = " + str(user_id)
    result = db.execute(query)
    return result

def process_items(items):
    result = []
    for item in items:
        if item > 10:
            result.append(item * 2)
    return result

"""

# 第一轮对话
response = llm.invoke(
     [
            {"role": "user", "content": user_prompt},
        ]
)
print("第一轮结果："+response.content)

print("---------------------------------------------------------")
