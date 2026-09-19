from openai import OpenAI
import json
import os

from dotenv import load_dotenv
load_dotenv()
client = OpenAI(
    base_url=os.environ["DEEPSEEK_API_URL"],
    api_key=os.environ["DEEPSEEK_API_KEY"],
)

# 1. 定义工具：ask_user
tools = [
    {
        "type": "function",
        "function": {
            "name": "ask_user",
            "description": "当缺少完成用户请求所必需的信息时，向用户提问以补充信息。",
            "parameters": {
                "type": "object",
                "properties": {"question": {"type": "string", "description": "要问用户的具体问题"}},
                "required": ["question"],
            },
        },
    }
]

messages = [
    {
        "role": "system",
        "content": "你是一个助手。如果用户提供的信息不足以完成任务，必须调用 ask_user 向用户提问。",
    },
    {"role": "user", "content": "想去常德石门有没有便宜的交通方式，预算不多"},  # 用户初始请求，信息不全
]

while True:
    # 2. 发起请求

    response = client.chat.completions.create(
        model=os.environ["DEEPSEEK_MODEL"],
        messages=messages,
        tools=tools,
    )

    msg = response.choices[0].message
    # 3. 检查是否有工具调用
    if msg.tool_calls:
        # 把模型的工具调用消息加入历史
        messages.append(msg)

        for tool_call in msg.tool_calls:
            if tool_call.function.name == "ask_user":
                args = json.loads(tool_call.function.arguments)

                print(f"需要补充的问题：{args}")
                question = args["question"]

                # 4. 向用户提问并获取输入
                print(f"Agent: {question}")
                user_answer = input("你: ")

                # 5. 把用户输入作为工具结果回传
                messages.append(
                    {"role": "tool", "tool_call_id": tool_call.id, "content": user_answer}
                )
        # 6. 继续循环，再次请求模型
        continue
    else:
        # 没有工具调用，模型给出了最终回复
        print(f"Agent: {msg.content}")
        break
