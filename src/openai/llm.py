import os

from dotenv import load_dotenv

from openai import OpenAI

load_dotenv()

llm = OpenAI(api_key=os.environ["DEEPSEEK_API_KEY"], base_url=os.environ["DEEPSEEK_API_URL"])

# response = llm.chat.completions.create(
#     model=os.environ["DEEPSEEK_MODEL"],
#     temperature=0.7,
#     messages=[{"role": "user", "content": "你好，我叫黄开,请记住我"}],
# )

# print(response.choices[0].message.content)

# response = llm.chat.completions.create(
#     model=os.environ["DEEPSEEK_MODEL"],
#     temperature=0.7,
#     messages=[
#         {"role": "user", "content": "你好，我叫黄开,请记住我"},
#         {"role": "assistant", "content": response.choices[0].message.content},
#         {"role": "user", "content": "我是谁？"},
#     ],
# )

# print(response.choices[0].message.content)


response = llm.chat.completions.create(
    model=os.environ["DEEPSEEK_MODEL"],
    temperature=0.7,
    messages=[{"role": "user", "content": "你好，我叫黄开,请记住我"}],
    stream=True,
)

for chunk in response:
    chat = chunk.choices[0].delta.content
    if chat:  # 判断chat是不是空值，是空值就不判断
        print(chat, end="")
