import json
import os
import re
from pathlib import Path
from typing import Any, Dict

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()
llm = ChatOpenAI(
    model=os.environ["DEEPSEEK_MODEL"],  # 真实模型名
    base_url=os.environ["DEEPSEEK_API_URL"],
    api_key=os.environ["DEEPSEEK_API_KEY"],  # 手动传 key（因为变量名不是 OPENAI_API_KEY）
)


# LLM 本质上生成的是文本序列，不保证输出合法的 JSON。
# 常见问题包括：
# 输出多余解释文字、
# JSON 语法错误、
# 字段名不一致、
# 类型不匹配（数字变成了字符串）、
# 嵌套层级混乱等。

# 结构化输出控制的核心目标是：确保模型输出 100% 可被 JSON.parse()
# 解析，且字段名、类型、结构都符合预定义规范。

# JSON 结构化返回策略有四个层面可以进行操作
# 1.prompt 格式指令，step1示例  ，
# 2.分隔符包装
# 3.json schema
# 4.llm  api 层面强制约束

# 如果你只是在网页对话里练习，用第一层 + 第二层就够了。
# 如果你写代码调 API，加上第四层最保险。
# 第三层 JSON Schema 在字段多、嵌套深时才需要。


# 练习

# 第一种
# Prompt 格式指令
system_prompt = """
请从以下文本中提取事件名称、时间、地点、人物、组织和关键数字，
只输出 JSON，只输出代码块，不要额外文字
"""


# 第二种
# Prompt 格式指令里面增加返回分隔符包装
# system_prompt = """
# 请从以下文本中提取事件名称、时间、地点、人物、组织和关键数字，
# 所有输出必须用 ```json 代码块包裹，只输出代码块，不要额外文字
# """
# 分隔符结果获取
# response = llm.invoke(
#     [
#         {"role": "user", "content": user_msg},
#         {"role": "system", "content": system_prompt},
#     ]
# )
# text = response.content
# # 提取代码块中的 JSON
# match = re.search(r"```json\s*(\{.*?\})\s*```", text, re.DOTALL)
# if match:
#     print(json.loads(match.group(1)))
# else:
# # 兜底
#     print(json.loads(text))


# 第三种
# Prompt 格式指令里面指定JSON Schema
# system_prompt = """
# 请严格按照以下 JSON Schema 定义的结构输出：

# {{
#   "type": "object",
#   "properties": {{
#     "事件名称": {{ "type": "string" }},
#     "时间": {{ "type": "string" }},
#     "人物": {{ "type": "array", "items": {{ "type": "string" }} }},
#     ...
#   }},
#   "required": ["事件名称", "时间", "人物", ...]
# }}

# """


# 第四种:
# api层强制约束,就是在大模型调用的时候对返回格式进行指定
#  response_format={{"type": "json_object"}}

# 第五种:
# api层强制约束,就是在大模型调用的时候使用instructor
# 1.定义一个对象
# 使用instructor.from_openai(openAI(),model=instructor.Model.JSON),得到一个client
# 通过client.chat.complations.create()里面指定返回对象就行


user_msg = """
2024年6月20日，华为在深圳总部举行了鸿蒙星河版发布会。
余承东宣布该系统已支持超过9000款应用，预计年底突破10000款。
当天股价上涨3.2%，市值增加约500亿元。
"""

response = llm.invoke(
    [
        {"role": "user", "content": user_msg},
        {"role": "system", "content": system_prompt},
    ]
)
text = response.content
print(text)


# 待补充项:
# 1.json schema 需要加强
