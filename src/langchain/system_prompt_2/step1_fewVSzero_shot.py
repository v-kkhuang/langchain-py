import os
from langchain_core import prompts
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()
llm = ChatOpenAI(
    model=os.environ["DEEPSEEK_MODEL"],  # 真实模型名
    base_url=os.environ["DEEPSEEK_API_URL"],
    api_key=os.environ["DEEPSEEK_API_KEY"],  # 手动传 key（因为变量名不是 OPENAI_API_KEY）
)


# Zero-shot（无示例）
# Zero-shot 输出可能的问题：字段名不统一（"事件名称" vs "event"）、缺少置信度、数字格式不一致、可能遗漏字段
# {
#   "事件名称": "鸿蒙星河版发布会",
#   "时间": "2024年6月20日",
#   "地点": "深圳总部",
#   "人物": ["余承东"],
#   "组织": ["华为"],
#   "关键数字": {
#     "已支持应用数量": 9000,
#     "预计年底应用数量": 10000,
#     "股价涨幅": "3.2%",
#     "市值增加": "约500亿元"
#   }
# }

# {
#   "事件名称": "鸿蒙星河版发布会",
#   "时间": "2024年6月20日",
#   "地点": "华为深圳总部",
#   "人物": ["余承东"],
#   "组织": ["华为"],
#   "关键数字": {
#     "已支持应用数": "超过9000款",
#     "年底预计应用数": "10000款",
#     "股价涨幅": "3.2%",
#     "市值增加": "约500亿元"
#   }
# }
system_prompt = """
请从以下文本中提取事件名称、时间、地点、人物、组织和关键数字，
以 JSON 格式输出
"""
user_msg = """
2024年6月20日，华为在深圳总部举行了鸿蒙星河版发布会。
余承东宣布该系统已支持超过9000款应用，预计年底突破10000款。
当天股价上涨3.2%，市值增加约500亿元。
"""


system_prompt = """
你是一个新闻信息提取助手。请从文本中提取以下字段：
事件名称、时间、地点、人物（数组）、组织（数组）、关键数字（数组）。
每个字段附带置信度（0-1）。不确定的字段填"未知"。

# 示例 1
输入：3月12日，苹果公司在加州库比蒂诺总部举办春季发布会，
蒂姆·库克展示了新款iPad Pro，起售价799美元，搭载M4芯片。
输出：
{{
  "事件名称": {{ "value": "苹果春季发布会", "confidence": 0.95 }},
  "时间": {{ "value": "3月12日", "confidence": 0.90 }},
  "地点": {{ "value": "加州库比蒂诺", "confidence": 0.95 }},
  "人物": {{ "value": ["蒂姆·库克"], "confidence": 0.95 }},
  "组织": {{ "value": ["苹果公司"], "confidence": 0.98 }},
  "关键数字": {{ "value": ["799美元", "M4芯片"], "confidence": 0.90 }}
}}

# 示例 2
输入：近日，某知名企业在一场内部活动中公布了新战略，
据传投入数十亿资金，但具体金额未披露。
输出：
{{
  "事件名称": {{ "value": "新战略公布", "confidence": 0.75 }},
  "时间": {{ "value": "未知", "confidence": 0.30 }},
  "地点": {{ "value": "未知", "confidence": 0.30 }},
  "人物": {{ "value": [], "confidence": 0.20 }},
  "组织": {{ "value": ["某知名企业"], "confidence": 0.60 }},
  "关键数字": {{ "value": ["数十亿"], "confidence": 0.70 }}
}}

"""


response = llm.invoke(
    [
        {"role": "user", "content": user_msg},
        {"role": "system", "content": system_prompt},
    ]
)

print(response.content)
