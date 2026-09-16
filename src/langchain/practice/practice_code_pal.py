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
你是一位高级软件工程师。请审查以下代码。

# PAL 要求：对每个发现的问题，写一段验证代码来证明问题存在
对每个发现的问题：
1. 先分析问题原因
2. 写一段 Python 验证代码来复现/证明这个问题
3. 给出修复方案和修复后的验证代码

# 待审查代码
def calculate_average(numbers):
    total = 0
    for i in range(len(numbers)):
        total += numbers[i]
    return total / len(numbers)

"""

# 第一轮对话
response = llm.invoke(
     [
            {"role": "user", "content": user_prompt},
        ]
)
print("第一轮结果："+response.content)

print("---------------------------------------------------------")
user_prompt = """

请检查你的审查报告，确认以下维度是否都已覆盖：
1. ✅ 功能 Bug（空输入、类型错误等）
2. ✅ 安全漏洞（注入、泄露等）
3. ✅ 性能问题（复杂度、内存）
4. ✅ 可维护性（命名、注释、风格）
5. ✅ 最佳实践（类型标注、异常处理）

如果有遗漏的维度，请补充。

"""
response = llm.invoke(
     [
            {"role": "assistant", "content": response.content},
            {"role": "user", "content": user_prompt},
        ]
)


print("第二轮结果："+response.content)
print("---------------------------------------------------------")


user_prompt = """

请将所有问题整理为以下统一格式的表格：

| 序号 | 严重程度 | 类别 | 问题描述 | 位置 | 修复建议 |
|------|---------|------|---------|------|---------|

并在表格下方给出修复后的完整代码。

"""
response = llm.invoke(
     [
            {"role": "assistant", "content": response.content},
            {"role": "user", "content": user_prompt},
        ]
)


print("第三轮结果："+response.content)
print("---------------------------------------------------------")



user_prompt = """

请对修复后的代码再做一次审查，确认：
1. 原来的问题是否都已修复
2. 修复过程是否引入了新问题
3. 修复后的代码是否还有改进空间

如果一切正常，输出 "审查通过"。

"""
response = llm.invoke(
     [
            {"role": "assistant", "content": response.content},
            {"role": "user", "content": user_prompt},
        ]
)


print("第四轮结果："+response.content)
print("---------------------------------------------------------")


