import json
import os

from dotenv import load_dotenv

from openai import OpenAI

load_dotenv()

# tools schema
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "查询天气",
            "parameters": {
                "type": "object",
                "propertites": {"city": {"type": "string", "description": "城市名称"}},
            },
            "required": ["city"],
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_temperature",
            "description": "查询气温",
            "parameters": {
                "type": "object",
                "propertites": {"city": {"type": "string", "description": "城市名称"}},
            },
            "required": ["city"],
        },
    },
]

messages = [{"role": "user", "content": "武汉的天气怎么样！多少气温呢"}]


# ---------- 2. 本地函数实现 ----------
def get_weather(city: str) -> str:
    if city == "武汉":
        return "非常好"
    if city == "北京":
        return "不太好"
    return "未知"


def get_temperature(city: str) -> str:
    if city == "武汉":
        return "26度"
    if city == "北京":
        return "8度"
    return "未知"


llm = OpenAI(base_url=os.environ["DEEPSEEK_API_URL"], api_key=os.environ["DEEPSEEK_API_KEY"])

response = llm.chat.completions.create(
    model=os.environ["DEEPSEEK_MODEL"], temperature=0.7, tools=tools, messages=messages
)
messages.append(response.choices[0].message)

tool_calls = response.choices[0].message.tool_calls

# 遍历ai返回的需要调用的多个工具集
if tool_calls:
    for tools_call in tool_calls:
        arguments = json.loads(tools_call.function.arguments)
        if tools_call.function.name == "get_weather":
            result = get_weather(**arguments)
        elif tools_call.function.name == "get_temperature":
            result = get_temperature(**arguments)
        else:
            result = "未知函数"
        messages.append({"role": "tool", "tool_call_id": tools_call.id, "content": result})


print(messages)
# 根据大模型的返回决定是否调用工具,将返回的结果拼接到message里面去，一起传给大模型总结：
response = llm.chat.completions.create(
    model=os.environ["DEEPSEEK_MODEL"], temperature=0.7, tools=tools, messages=messages
)
print(response.choices[0].message.content)
