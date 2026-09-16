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


# 简单的系统提示词，直接使用 system
system_prompt = """
你是一个新闻信息提取助手。只输出 JSON，不要其他文字。

规则：
1. 每个字段包含 value 和 confidence（0-1）
2. 信息明确 → confidence 0.9+，模糊 → 0.5-0.7，未提及 → value 填 "未知"，confidence 填 0.1
3. 不要编造信息

--- 示例 1 ---
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

--- 示例 2 ---
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

# 完整测试
# user_prompt = """
# 2024年9月10日，苹果公司在加州库比蒂诺的史蒂夫·乔布斯剧院举行了秋季产品发布会。CEO蒂姆·库克发布了iPhone 16系列，起售价699美元，搭载A18 Pro芯片。首席财务官卢卡·梅斯特里表示，预计第四季度营收将达到890亿美元至910亿美元之间。

# """

# 结果:
# {
#   "事件名称": {"value": "秋季产品发布会", "confidence": 0.95},
#   "时间": {"value": "2024年9月10日", "confidence": 0.95},
#   "地点": {"value": "加州库比蒂诺的史蒂夫·乔布斯剧院", "confidence": 0.95},
#   "人物": {"value": ["蒂姆·库克", "卢卡·梅斯特里"], "confidence": 0.95},
#   "组织": {"value": ["苹果公司"], "confidence": 0.98},
#   "关键数字": {"value": ["699美元", "890亿美元至910亿美元"], "confidence": 0.90}
# }


# 异常测试
user_prompt = """
近日，某国际环保组织发布报告称，全球海洋塑料污染情况日益严重。报告指出，每年约有数百万吨塑料废物流入海洋，对海洋生态造成不可逆转的损害。该组织呼吁各国政府采取紧急措施。
"""
# 结果:
# {
#   "事件名称": { "value": "国际环保组织发布全球海洋塑料污染报告", "confidence": 0.8 },
#   "时间": { "value": "近日", "confidence": 0.6 },
#   "地点": { "value": "未知", "confidence": 0.1 },
#   "人物": { "value": [], "confidence": 0.1 },
#   "组织": { "value": ["某国际环保组织"], "confidence": 0.6 },
#   "关键数字": { "value": ["数百万吨"], "confidence": 0.65 }
# }


response = llm.invoke(
    [
        {"role": "user", "content": user_prompt},
        {"role": "system", "content": system_prompt},
    ]
)

print(response.content)
