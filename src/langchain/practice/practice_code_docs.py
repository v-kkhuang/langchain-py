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
你是一位技术文档工程师。

生成文档流程：
1. 先分析函数的输入、处理逻辑、输出
2. 识别边界情况和异常条件
3. 生成标准 docstring（Google 风格）
4. 为复杂逻辑添加行内注释

--- 示例 ---
代码：
def binary_search(arr, target):
    left, right = 0, len(arr) - 1
    while left <= right:
        mid = (left + right) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    return -1

文档：
使用二分查找在有序数组中搜索目标值。

    Args:
        arr: 已排序的数字列表（升序）。
        target: 要查找的目标值。

    Returns:
        int: 目标值的索引，找不到返回 -1。

    Raises:
        TypeError: arr 包含非数字元素时可能出错。

    Time Complexity: O(log n)
    Space Complexity: O(1)

    Example:
        >>> binary_search([1, 3, 5, 7, 9], 5)
        2
        >>> binary_search([1, 3, 5, 7, 9], 4)
        -1
def binary_search(arr, target):
    left, right = 0, len(arr) - 1
    while left <= right:
        mid = (left + right) // 2  # 避免溢出：left + (right - left) // 2 更安全
        ...

--- 请为以下代码生成文档 ---
代码：


class TaskQueue:
    def __init__(self, max_size=100):
        self.queue = []
        self.max_size = max_size

    def push(self, task):
        if len(self.queue) >= self.max_size:
            raise OverflowError("队列已满")
        self.queue.append(task)

    def pop(self):
        if not self.queue:
            return None
        return self.queue.pop(0)

    def size(self):
        return len(self.queue)

"""

# 第一轮对话
response = llm.invoke(
     [
            {"role": "user", "content": user_prompt},
        ]
)
print("第一轮结果："+response.content)

print("---------------------------------------------------------")
