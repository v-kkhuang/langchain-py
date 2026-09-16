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

# 简单的系统提示词，直接使用 system
system_prompt = """
【系统指令】 定义角色和任务描述
你是一个信息提取助手。请从用户提供的文本中
提取关键信息，并以 JSON 格式输出。

【格式说明】 定义输出结构
输出格式要求：
{{
  "字段1": "值（字符串）",
  "字段2": "值（数字）",
  "confidence": "置信度（0-1）"
}}

【示例 1】 输入-输出对
输入：张三于2024年3月15日在北京参加了马拉松比赛。
输出：
{{
  "事件": "马拉松比赛",
  "时间": "2024年3月15日",
  "地点": "北京",
  "人物": ["张三"],
  "confidence": 0.95
}}

【示例 2】 另一个输入-输出对
输入：李四和王五昨天在上海发布了新产品。
输出：
{{
  "事件": "新产品发布",
  "时间": "未知",
  "地点": "上海",
  "人物": ["李四", "王五"],
  "confidence": 0.80
}}


"""


response = llm.invoke(
    [
        {
            "role": "user",
            "content": "外交部发言人郭嘉昆主持例行记者会。\
新加坡亚洲新闻台记者提问，新加坡外交部刚刚公布消息，52名新加坡公民在中国广西被捕，他们疑似从事传销活动",
        },
        {"role": "system", "content": system_prompt},
    ]
)

print(response.content)


# 示例要有多样性：覆盖不同场景——信息完整的、信息缺失的、信息模糊的。这教会模型在缺失时标注"未知"。
# 示例要展示边界情况：比如时间不明确时输出"未知"，有多个人物时输出数组，有数字时提取准确数值。
# 示例数量适中：2-3 个高质量示例通常优于 5 个平庸示例。太多示例会占用 Token 且可能引入噪声。
# 示例顺序有讲究：将最典型的示例放在最后（最接近实际输入的位置），因为模型对最近的示例记忆最强。
# 置信度要有区分度：信息明确的示例设高置信度（0.9+），模糊的设低置信度（0.5-0.7），教会模型自我评估。
