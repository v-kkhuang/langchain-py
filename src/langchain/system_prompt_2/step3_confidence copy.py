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


# user_msg = """
# 2024年6月20日，华为在深圳总部举行了鸿蒙星河版发布会。
# 余承东宣布该系统已支持超过9000款应用，预计年底突破10000款。
# 当天股价上涨3.2%，市值增加约500亿元。
# """


user_msg = """
近日，某知名企业在一场内部活动中公布了新战略，据传投入数十亿资金，但具体金额未披露
"""


system_prompt = """
你是一个新闻信息提取助手。只输出 JSON，不要其他文字。

规则：
1. 每个字段包含 value（值）和 confidence（置信度 0-1）
2. 信息明确提及 → confidence 0.9 以上
3. 信息模糊或不精确 → confidence 0.5-0.7
4. 文本中未提及 → value 填 "未知"，confidence 填 0.1
5. 不要编造原文中不存在的信息

--- 示例 1（信息完整）---
输入：3月12日，苹果公司在加州库比蒂诺总部举办春季发布会，蒂姆·库克展示了新款iPad Pro，起售价799美元。
输出：
{{
  "事件名称": {{ "value": "苹果春季发布会", "confidence": 0.95 }},
  "时间": {{ "value": "3月12日", "confidence": 0.90 }},
  "地点": {{ "value": "加州库比蒂诺", "confidence": 0.95 }},
  "人物": {{ "value": ["蒂姆·库克"], "confidence": 0.95 }},
  "组织": {{ "value": ["苹果公司"], "confidence": 0.98 }},
  "关键数字": {{ "value": ["799美元"], "confidence": 0.90 }}
}}

--- 示例 2（信息缺失）---
输入：近日，某企业在内部活动中公布了新战略，据传投入数十亿资金。
输出：
{{
  "事件名称": {{ "value": "新战略公布", "confidence": 0.75 }},
  "时间": {{ "value": "未知", "confidence": 0.10 }},
  "地点": {{ "value": "未知", "confidence": 0.10 }},
  "人物": {{ "value": [], "confidence": 0.10 }},
  "组织": {{ "value": ["某企业"], "confidence": 0.60 }},
  "关键数字": {{ "value": ["数十亿"], "confidence": 0.65 }}
}}
"""


response = llm.invoke(
    [
        {"role": "user", "content": user_msg},
        {"role": "system", "content": system_prompt},
    ]
)

print(response.content)
