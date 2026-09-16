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

# 关键：要求逐步思考（Chain-of-Thought）
请按以下步骤审查代码：
1. 先逐行阅读代码，理解每一行的功能和数据流
2. 追踪变量从输入到输出的变化过程
3. 思考边界情况（空输入、极大值、负数、非法类型）
4. 检查安全问题（注入、泄露、权限）
5. 分析性能（时间复杂度、内存）
6. 最后汇总所有发现，按严重程度排列

# 待审查代码
def get_user_data(user_id):
    query = "SELECT * FROM users WHERE id = " + str(user_id)
    result = db.execute(query)
    return result

"""


response = llm.invoke(
     [
            {"role": "user", "content": user_prompt},
        ]
)
print(response.content)

