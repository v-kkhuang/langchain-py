import json
import os

from dotenv import load_dotenv

from openai import OpenAI


load_dotenv()

llm = OpenAI(base_url=os.environ["DEEPSEEK_API_URL"], api_key=os.environ["DEEPSEEK_API_KEY"])


WEATHER_DB = {
    "北京": {"celsius": 22, "fahrenheit": 72, "condition": "晴", "humidity": 45},
    "上海": {"celsius": 26, "fahrenheit": 79, "condition": "多云", "humidity": 68},
    "广州": {"celsius": 30, "fahrenheit": 86, "condition": "雷阵雨", "humidity": 82},
    "深圳": {"celsius": 29, "fahrenheit": 84, "condition": "晴", "humidity": 75},
    "成都": {"celsius": 24, "fahrenheit": 75, "condition": "阴", "humidity": 70},
}

RESTAURANT_DB = {
    "北京": [
        {"name": "全聚德", "cuisine": "中餐", "price": 200, "rating": 4.5},
        {"name": "海底捞", "cuisine": "火锅", "price": 150, "rating": 4.7},
        {"name": "大董烤鸭", "cuisine": "中餐", "price": 300, "rating": 4.6},
        {"name": "京味斋", "cuisine": "京菜", "price": 100, "rating": 4.3},
    ],
    "上海": [
        {"name": "南翔小笼", "cuisine": "中餐", "price": 80, "rating": 4.6},
        {"name": "M on the Bund", "cuisine": "西餐", "price": 500, "rating": 4.8},
        {"name": "海底捞", "cuisine": "火锅", "price": 160, "rating": 4.7},
        {"name": "老正兴", "cuisine": "沪菜", "price": 180, "rating": 4.4},
    ],
    "广州": [
        {"name": "陶陶居", "cuisine": "粤菜", "price": 120, "rating": 4.7},
        {"name": "炳胜品味", "cuisine": "粤菜", "price": 200, "rating": 4.6},
        {"name": "海底捞", "cuisine": "火锅", "price": 140, "rating": 4.5},
        {"name": "陈添记", "cuisine": "小吃", "price": 50, "rating": 4.8},
    ],
}

UNIT = "celsius"


def get_weather(city: str):
    if city not in WEATHER_DB:
        return {"error": f"暂不支持{city}，查询天气！"}
    data = WEATHER_DB[city]
    return {
        "city": city,
        "temperature": data["celsius"] if UNIT == "celsius" else data["fahrenheit"],
        "unit": UNIT,
        "condition": data["condition"],
        "humidity": data["humidity"],
    }


def search_restaurants(city, cuisine="", max_price=0):
    if city not in RESTAURANT_DB:
        return {"error": f"暂不支持查询「{city}」的餐厅"}
    results = RESTAURANT_DB[city]
    if cuisine:
        results = [r for r in results if cuisine == r["cuisine"]]
    if max_price > 0:
        results = [r for r in results if r["price"] <= max_price]
    return {"city": city, "count": len(results), "restaurants": results}


tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "获取天气信息",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string", "description": "城市名称"}},
            },
            "required": ["city"],
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_restaurants",
            "description": "获取餐厅信息",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "城市名称"},
                    "cuisine": {"type": "string", "description": "菜系"},
                    "max_price": {"type": "number", "description": "预算"},
                },
            },
            "required": ["city", "cuisine", "max_price"],
        },
    },
]


messages = [{"role": "user", "content": "石门有什么火锅店推荐？喜欢吃老北京铜锅"}]

response = llm.chat.completions.create(
    model=os.environ["DEEPSEEK_MODEL"], temperature=0.7, messages=messages, tools=tools
)


print(f"大模型工具推荐：{response.choices[0].message.tool_calls}")

messages.append(response.choices[0].message)
tools_calls = response.choices[0].message.tool_calls
if tools_calls:
    for tool_call in tools_calls:
        tools_name = tool_call.function.name
        arg = json.loads(tool_call.function.arguments)
        if tools_name == "get_weather":
            result = get_weather(**arg)
        elif tools_name == "search_restaurants":
            result = search_restaurants(**arg)
        else:
            result = "未知函数"
        messages.append(
            {
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": json.dumps(result, ensure_ascii=False),
            }
        )

print("\n")
print(f"拼接最后的message：{messages}")
print("\n")
# 根据大模型的返回决定是否调用工具,将返回的结果拼接到message里面去，一起传给大模型总结：
response = llm.chat.completions.create(
    model=os.environ["DEEPSEEK_MODEL"], temperature=0.7, tools=tools, messages=messages
)
print(f"llm回复：{response.choices[0].message.content}")
