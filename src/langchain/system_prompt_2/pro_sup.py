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
你是一个产品评论分析助手。只输出 JSON，不要其他文字。

规则：
1. 每个字段包含 value 和 confidence（0-1）
2. 评分 1-5：5=非常满意，4=满意有小瑕疵，3=褒贬不一，2=不满意，1=非常不满
3. 购买建议："推荐" / "不推荐" / "看情况"
4. 优点和缺点从原文提取，不要编造
5. 未提及产品名称填 "未知"

--- 示例 1（正面评论）---
输入：买了这款索尼WH-1000XM5耳机，降噪效果绝了！戴上后地铁噪音几乎消失。音质低频饱满。就是价格有点贵2899元，长时间佩戴有压迫感。总体推荐！
输出：
{{
  "产品名称": {{ "value": "索尼WH-1000XM5", "confidence": 0.98 }},
  "评分": {{ "value": 4, "confidence": 0.85 }},
  "优点": {{ "value": ["降噪出色", "音质低频饱满"], "confidence": 0.95 }},
  "缺点": {{ "value": ["价格贵2899元", "长时间佩戴有压迫感"], "confidence": 0.92 }},
  "购买建议": {{ "value": "推荐", "confidence": 0.90 }}
}}

--- 示例 2（负面评论）---
输入：这个手机壳用了一周就发黄了，质量太差了！不推荐。
输出：
{{
  "产品名称": {{ "value": "未知", "confidence": 0.10 }},
  "评分": {{ "value": 1, "confidence": 0.90 }},
  "优点": {{ "value": [], "confidence": 0.10 }},
  "缺点": {{ "value": ["一周发黄", "质量差"], "confidence": 0.90 }},
  "购买建议": {{ "value": "不推荐", "confidence": 0.95 }}
}}

"""

# 完整测试
# user_prompt = """
# 小米14 Pro入手两周，体验非常好！徕卡光学镜头拍出来的照片色彩好看，夜景模式暗光也清晰。骁龙8 Gen3性能强劲，玩原神全高画质60帧不卡。90W快充半小时充满。唯一遗憾是没有无线充电。总体非常推荐！
# """

# 结果:
# {{
#   "产品名称": {{ "value": "小米14 Pro", "confidence": 0.95 }},
#   "评分": {{ "value": 4, "confidence": 0.80 }},
#   "优点": {{ "value": ["拍照色彩好看", "夜景模式暗光清晰", "性能强劲", "90W快充半小时充满"], "confidence": 0.93 }},
#   "缺点": {{ "value": ["没有无线充电"], "confidence": 0.90 }},
#   "购买建议": {{ "value": "推荐", "confidence": 0.92 }}
# }}


# 异常测试
user_prompt = """
戴森V12用了三个月来评价。吸力确实不错，地毯灰尘都能吸干净，这是优点。但续航太短了，大面积打扫得中途充电，很不方便。而且噪音很大。价格也不便宜，四千多买的。要说推荐不推荐...看个人需求吧，小户型可以，大房子就算了。
# """
# 结果:
# {{
#   "产品名称": {{ "value": "戴森V12", "confidence": 0.98 }},
#   "评分": {{ "value": 3, "confidence": 0.85 }},
#   "优点": {{ "value": ["吸力不错", "地毯灰尘能吸干净"], "confidence": 0.95 }},
#   "缺点": {{ "value": ["续航太短，大面积需中途充电", "噪音很大", "价格贵，四千多"], "confidence": 0.92 }},
#   "购买建议": {{ "value": "看情况", "confidence": 0.90 }}
# }}


response = llm.invoke(
    [
        {"role": "user", "content": user_prompt},
        {"role": "system", "content": system_prompt},
    ]
)

print(response.content)
