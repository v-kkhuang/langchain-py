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
你是一位测试工程师。

生成测试用例流程：
1. 分析函数的输入参数和返回值
2. 设计正常用例、边界用例、异常用例
3. 生成可运行的 pytest 代码
4. 每个测试函数有清晰的命名和注释

测试类型必须覆盖：
- ✅ 正常输入（典型值）
- ✅ 边界值（空、零、最大值、最小值）
- ✅ 异常输入（None、错误类型）
- ✅ 特殊情况（负数、重复值）

--- 示例 ---
代码：
def clamp(value, min_val, max_val):
    if value < min_val:
        return min_val
    if value > max_val:
        return max_val
    return value

测试：
import pytest

def test_clamp_normal():
    ""正常值不被截断""
    assert clamp(5, 0, 10) == 5

def test_clamp_below_min():
    ""低于最小值时截断""
    assert clamp(-5, 0, 10) == 0

def test_clamp_above_max():
    ""高于最大值时截断""
    assert clamp(15, 0, 10) == 10

def test_clamp_boundary():
    ""边界值本身不被截断""
    assert clamp(0, 0, 10) == 0
    assert clamp(10, 0, 10) == 10

def test_clamp_equal_min_max():
    ""min == max 的特殊情况""
    assert clamp(5, 7, 7) == 7

--- 请为以下代码生成测试 ---

def merge_sorted_lists(list1, list2):
    merged = []
    i, j = 0, 0
    while i < len(list1) and j < len(list2):
        if list1[i] <= list2[j]:
            merged.append(list1[i])
            i += 1
        else:
            merged.append(list2[j])
            j += 1
    merged.extend(list1[i:])
    merged.extend(list2[j:])
    return merged

"""

# 第一轮对话
response = llm.invoke(
     [
            {"role": "user", "content": user_prompt},
        ]
)
print("第一轮结果："+response.content)

print("---------------------------------------------------------")
