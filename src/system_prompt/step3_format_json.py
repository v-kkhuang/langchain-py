import os

import instructor
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from openai import OpenAI
from pydantic import BaseModel, Field

load_dotenv()
llm = ChatOpenAI(
    model=os.environ["DEEPSEEK_MODEL"],  # 真实模型名
    base_url=os.environ["DEEPSEEK_API_URL"],
    api_key=os.environ["DEEPSEEK_API_KEY"],  # 手动传 key（因为变量名不是 OPENAI_API_KEY）
)


# 方式一： 通过提示词prompt限制

# prompt = (
#     "请以 JSON 格式输出摘要，严格遵循以下结构，不要输出任何其他内容:"
#     "{'ummary': '摘要正文', 'key_points': ['要点1', '要点2'], 'word_count': 85}"
# )

prompt = (
    "请以 JSON 格式输出摘要:"
    "{'ummary': '摘要正文', 'key_points': ['要点1', '要点2'], 'word_count': 85}"
)

text = """
人工智能（AI）是计算机科学的一个分支，致力于创造能够模拟人类智能行为的系统。
近年来，深度学习技术的突破使得AI在图像识别、自然语言处理、语音识别等领域取得了显著进展。
然而，AI的发展也带来了伦理挑战，包括隐私保护、算法偏见、就业影响等问题。
专家认为，未来AI的发展需要在技术创新和伦理约束之间找到平衡。
"""

# response = llm.invoke(
#     [
#         {"role": "user", "content": text},
#         {"role": "system", "content": prompt},
#     ]
# )
# print(response.content)


# {'summary': '人工智能作为计算机科学分支，旨在模拟人类智能。深度学习推动其在图像识别、自然语言处理等领域取得显著进展，但同时也引发隐私、偏见、就业等伦理挑战。未来需在创新与伦理间寻求平衡。', 'key_points': ['AI是模拟人类智能的计算机科学分支', '深度学习促进多领域技术突破', 'AI发展引发隐私、偏见、就业等伦理问题', '未来需平衡技术创新与伦理约束'], 'word_count': 85}


#  下面是jsong格式：{'summary': '人工智能作为计算机科学分支，旨在模拟人类智能。深度学习推动其在图像识别、自然语言处理等领域取得显著进展，但同时也引发隐私、偏见、就业等伦理挑战。未来需在创新与伦理间寻求平衡。', 'key_points': ['AI是模拟人类智能的计算机科学分支', '深度学习促进多领域技术突破', 'AI发展引发隐私、偏见、就业等伦理问题', '未来需平衡技术创新与伦理约束'], 'word_count': 85}

# 总结：
# 通过提示词让大模型返回json，不够严谨，
# 后续任务代码中用 json.loads() 解析。缺点是模型偶尔会输出多余文字导致 JSON 解析失败。


# 方式二： 通过Instructor库：Instructor 基于 Pydantic 模型，能让 API 直接返回结构化 Python 对象


# 使用instructor包装 client
# 注意：deepseek-v4-flash 是思维链模型，不支持 tool_choice（函数调用），
# 所以这里用 Mode.JSON，让 instructor 走 response_format=json_object 而不是工具调用。
client = instructor.from_openai(
    OpenAI(base_url=os.environ["DEEPSEEK_API_URL"], api_key=os.environ["DEEPSEEK_API_KEY"]),
    mode=instructor.Mode.JSON,
)


# 定义返回对象格式
class SummaryResult(BaseModel):
    summary: str = Field(description="摘要正文")
    key_points: list[str] = Field(description="3-5个关键要点")
    word_count: int = Field(description="摘要正文字数")


# 调用llm


def summarize_structured(text: str) -> SummaryResult:
    return client.chat.completions.create(
        model=os.environ["DEEPSEEK_MODEL"],  # 真实模型名
        response_model=SummaryResult,
        messages=[
            {"role": "system", "content": "你是文本摘要专家。"},
            {"role": "user", "content": f"请总结：\n\n{text}"},
        ],
    )


# 直接拿到 Python 对象
result = summarize_structured(text)
print(result.summary)
print(result.key_points)
print(result.word_count)
print(result)


# 遇到的问题

# 你的 DEEPSEEK_MODEL=deepseek-v4-flash 是思维链（thinking）模型。Instructor 默认用 function calling
# （即发送 tool_choice 参数）来约束结构化输出，而 DeepSeek 的思维链模型不支持 tool_choice，所以返回 400 错误：
# 'Thinking mode does not support this tool_choice'


# 返回结果：
# 人工智能是计算机科学的分支，旨在模拟人类智能。深度学习推动其在图像识别、自然语言处理等领域取得显著进展，但也带来隐私、偏见和就业等伦理挑战。未来需在创新与伦理间寻求平衡。
# ['人工智能是计算机科学中模拟人类智能的分支', '深度学习促进AI在多个领域取得重大进展', 'AI发展引发隐私保护、算法偏见和就业影响等伦理问题', '未来AI需要在技术创新与伦理约束间保持平衡']
# 74
# summary='人工智能是计算机科学的分支，旨在模拟人类智能。深度学习推动其在图像识别、自然语言处理等领域取得显著进展，但也带来隐私、偏见和就业等伦理挑战。未来需在创新与伦理间寻求平衡。' key_points=['人工智能是计算机科学中模拟人类智能的分支', '深度学习促进AI在多个领域取得重大进展', 'AI发展引发隐私保护、算法偏见和就业影响等伦理问题', '未来AI需要在技术创新与伦理约束间保持平衡'] word_count=74
