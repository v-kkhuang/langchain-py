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
你是一个简历信息提取助手。只输出 JSON，不要其他文字。

规则：
1. 每个字段包含 value 和 confidence（0-1）
2. 未提及的字段 value 填 "未知"，confidence 填 0.1
3. 不要编造信息
4. 教育和工作经历按时间倒序（最近的在前）

--- 示例 ---
输入：
张三
电话：13800138000 | 邮箱：zhangsan@email.com | 北京
教育背景：
2018-2022 清华大学 计算机科学与技术 本科
工作经历：
2022.07-至今 字节跳动 后端开发工程师，负责微服务架构设计
2021.06-2021.09 腾讯 实习生，参与后端开发
技能：Python, Go, Docker, MySQL

输出：
{{
  "姓名": {{ "value": "张三", "confidence": 0.99 }},
  "联系方式": {{
    "value": {{
      "电话": "13800138000",
      "邮箱": "zhangsan@email.com",
      "城市": "北京"
    }},
    "confidence": 0.98
  }},
  "教育背景": {{
    "value": [
      {{ "学校": "清华大学", "学位": "本科", "专业": "计算机科学与技术", "时间段": "2018-2022" }}
    ],
    "confidence": 0.98
  }},
  "工作经历": {{
    "value": [
      {{ "公司": "字节跳动", "职位": "后端开发工程师", "时间段": "2022.07-至今", "内容": "负责微服务架构设计" }},
      {{ "公司": "腾讯", "职位": "实习生", "时间段": "2021.06-2021.09", "内容": "参与后端开发" }}
    ],
    "confidence": 0.97
  }},
  "技能": {{ "value": ["Python", "Go", "Docker", "MySQL"], "confidence": 0.98 }}
}}

"""

# 完整测试
# user_prompt = """
# 李四
# 手机：13912345678 | 邮箱：lisi@gmail.com | 上海浦东新区

# 教育经历：
# 2016.09 - 2019.06  浙江大学  软件工程  硕士
# 2012.09 - 2016.06  武汉大学  软件工程  本科

# 工作经历：
# 2023.01 - 至今  阿里巴巴  高级算法工程师
#   负责推荐系统算法优化，提升CTR 15%
# 2019.07 - 2022.12  百度  算法工程师
#   参与搜索引擎排序算法研发

# 专业技能：
# 机器学习、深度学习、TensorFlow、PyTorch、C++、Java
# """

# 结果:
# {{
#   "姓名": {{ "value": "李四", "confidence": 0.99 }},
#   "联系方式": {{
#     "value": {{
#       "电话": "13912345678",
#       "邮箱": "lisi@gmail.com",
#       "城市": "上海浦东新区"
#     }},
#     "confidence": 0.98
#   }},
#   "教育背景": {{
#     "value": [
#       {{ "学校": "浙江大学", "学位": "硕士", "专业": "软件工程", "时间段": "2016.09 - 2019.06" }},
#       {{ "学校": "武汉大学", "学位": "本科", "专业": "软件工程", "时间段": "2012.09 - 2016.06" }}
#     ],
#     "confidence": 0.98
#   }},
#   "工作经历": {{
#     "value": [
#       {{ "公司": "阿里巴巴", "职位": "高级算法工程师", "时间段": "2023.01 - 至今", "内容": "负责推荐系统算法优化，提升CTR 15%" }},
#       {{ "公司": "百度", "职位": "算法工程师", "时间段": "2019.07 - 2022.12", "内容": "参与搜索引擎排序算法研发" }}
#     ],
#     "confidence": 0.97
#   }},
#   "技能": {{ "value": ["机器学习", "深度学习", "TensorFlow", "PyTorch", "C++", "Java"], "confidence": 0.98 }}
# }}


# 异常测试
user_prompt = """
王五

意向职位：数据分析师

学历：本科
毕业院校：某重点大学统计学专业

工作经历：
- 2020年至今：在某互联网公司做数据分析相关的工作
- 之前还有几段实习经历

技能：Excel, SQL, 有一定的Python基础"""
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
