import json
import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

llm = OpenAI(
    base_url=os.environ["DEEPSEEK_API_URL"],
    api_key=os.environ["DEEPSEEK_API_KEY"],
)

# ============================================================
# 数据层：故意只覆盖部分城市/菜系，制造"查不到"的场景
# ============================================================
WEATHER_DB = {
    "北京": {"celsius": 22, "condition": "晴", "humidity": 45},
    "上海": {"celsius": 26, "condition": "多云", "humidity": 68},
    "广州": {"celsius": 30, "condition": "雷阵雨", "humidity": 82},
}

RESTAURANT_DB = {
    "北京": [
        {"name": "全聚德", "cuisine": "中餐", "price": 200, "rating": 4.5},
        {"name": "海底捞", "cuisine": "火锅", "price": 150, "rating": 4.7},
    ],
    "上海": [
        {"name": "南翔小笼", "cuisine": "中餐", "price": 80, "rating": 4.6},
        {"name": "海底捞", "cuisine": "火锅", "price": 160, "rating": 4.7},
    ],
}

HOTEL_DB = {
    "北京": [{"name": "王府井饭店", "price": 800, "rating": 4.5}],
    "上海": [{"name": "和平饭店", "price": 1200, "rating": 4.8}],
}

FLIGHT_DB = {
    ("北京", "上海"): [{"flight": "CA1501", "price": 1200, "depart": "08:00"}],
    ("上海", "北京"): [{"flight": "MU5102", "price": 1150, "depart": "09:30"}],
}

TRAIN_DB = {
    ("北京", "上海"): [{"train": "G1", "price": 553, "depart": "09:00"}],
    ("上海", "北京"): [{"train": "G2", "price": 553, "depart": "09:00"}],
}


# ============================================================
# 工具函数：故意不做严格的参数归一化
# ============================================================
def get_weather(city: str):
    if city not in WEATHER_DB:
        return {"error": f"暂不支持{city}的天气查询"}
    return {"city": city, **WEATHER_DB[city]}


def search_restaurants(city: str, cuisine: str = "", max_price: float = 0):
    if city not in RESTAURANT_DB:
        return {"error": f"暂不支持查询「{city}」的餐厅"}
    results = RESTAURANT_DB[city]
    if cuisine:
        results = [r for r in results if cuisine == r["cuisine"]]
    if max_price > 0:
        results = [r for r in results if r["price"] <= max_price]
    return {"city": city, "count": len(results), "restaurants": results}


def search_hotels(city: str, max_price: float = 0):
    if city not in HOTEL_DB:
        return {"error": f"暂不支持查询「{city}」的酒店"}
    results = HOTEL_DB[city]
    if max_price > 0:
        results = [r for r in results if r["price"] <= max_price]
    return {"city": city, "count": len(results), "hotels": results}


def search_flights(from_city: str, to_city: str):
    key = (from_city, to_city)
    if key not in FLIGHT_DB:
        return {"error": f"暂不支持查询「{from_city}」到「{to_city}」的航班"}
    return {"from": from_city, "to": to_city, "flights": FLIGHT_DB[key]}


def search_trains(from_city: str, to_city: str):
    key = (from_city, to_city)
    if key not in TRAIN_DB:
        return {"error": f"暂不支持查询「{from_city}」到「{to_city}」的火车"}
    return {"from": from_city, "to": to_city, "trains": TRAIN_DB[key]}


def get_exchange_rate(from_currency: str, to_currency: str):
    rates = {("USD", "CNY"): 7.2, ("CNY", "USD"): 0.139, ("EUR", "CNY"): 7.8}
    key = (from_currency, to_currency)
    if key not in rates:
        return {"error": f"暂不支持{from_currency}到{to_currency}的汇率"}
    return {"from": from_currency, "to": to_currency, "rate": rates[key]}


# ============================================================
# 工具定义：故意写得含糊、参数没有 enum 约束
# 让 DeepSeek 容易选错工具、填错参数
# ============================================================
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "查询天气",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "enum": ["北京", "上海", "广州"],
                        "description": "要查询的城市，必须是以下之一：北京、上海、广州",
                    }
                },
                "required": ["city"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_restaurants",
            "description": "查询餐厅",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "enum": ["北京", "上海", "广州"],
                        "description": "要查询的城市，必须是以下之一：北京、上海、广州",
                    },
                    "cuisine": {"type": "string", "description": "菜系"},
                    "max_price": {"type": "number", "description": "价格"},
                },
                "required": ["city", "cuisine", "max_price"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_hotels",
            "description": "查询酒店",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "enum": ["北京", "上海", "广州"],
                        "description": "要查询的城市，必须是以下之一：北京、上海、广州",
                    },
                    "max_price": {"type": "number", "description": "价格"},
                },
                "required": ["city"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_flights",
            "description": "查询航班信息，在用户查询航班信息时或者飞机相关信息时使用此工具，\
                            用户查询其他交通工具的时候别使用此工具查询",
            "parameters": {
                "type": "object",
                "properties": {
                    "from_city": {"type": "string", "description": "出发地"},
                    "to_city": {"type": "string", "description": "目的地"},
                },
                "required": ["from_city", "to_city"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_trains",
            "description": "查询火车，高铁信息，在用户查询火车，高铁信息时相关信息时使用此工具，\
                               用户查询其他交通工具的时候别使用此工具查询",
            "parameters": {
                "type": "object",
                "properties": {
                    "from_city": {"type": "string", "description": "出发地"},
                    "to_city": {"type": "string", "description": "目的地"},
                },
                "required": ["from_city", "to_city"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_exchange_rate",
            "description": "查询汇率",
            "parameters": {
                "type": "object",
                "properties": {
                    "from_currency": {"type": "string", "description": "源货币"},
                    "to_currency": {"type": "string", "description": "目标货币"},
                },
                "required": ["from_currency", "to_currency"],
            },
        },
    },
]

# ============================================================
# 用户问题：故意包含"交通""预算""旧称""细分菜系"等模糊信息
# ============================================================
messages = [
    {
        "role": "user",
        "content": "我下个月想去石门吃老北京铜锅，顺便看看有没有便宜的交通方式，预算不多。",
    }
]

# ============================================================
# 第一轮：让 DeepSeek 自己决定调哪些工具
# ============================================================
response = llm.chat.completions.create(
    model=os.environ["DEEPSEEK_MODEL"],
    temperature=0.7,
    messages=messages,
    tools=tools,
)

print("=" * 60)
print("【第一轮 tool_calls】")
print(response.choices[0].message.tool_calls)
print("=" * 60)

messages.append(response.choices[0].message)
tool_calls = response.choices[0].message.tool_calls

# ============================================================
# 执行工具：故意只执行一次，不处理模型可能的"自我纠正"
# ============================================================
if tool_calls:
    for tool_call in tool_calls:
        name = tool_call.function.name
        args = json.loads(tool_call.function.arguments)
        print(f"\n[执行] {name}({args})")

        if name == "get_weather":
            result = get_weather(**args)
        elif name == "search_restaurants":
            result = search_restaurants(**args)
        elif name == "search_hotels":
            result = search_hotels(**args)
        elif name == "search_flights":
            result = search_flights(**args)
        elif name == "search_trains":
            result = search_trains(**args)
        elif name == "get_exchange_rate":
            result = get_exchange_rate(**args)
        else:
            result = {"error": "未知函数"}

        print(f"[结果] {json.dumps(result, ensure_ascii=False)}")

        messages.append(
            {
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": json.dumps(result, ensure_ascii=False),
            }
        )

# ============================================================
# 第二轮：模型基于工具结果总结
# ============================================================
final_response = llm.chat.completions.create(
    model=os.environ["DEEPSEEK_MODEL"],
    temperature=0.7,
    messages=messages,
    tools=tools,
)

print("\n" + "=" * 60)
print("【最终回复】")
print(final_response.choices[0].message.content)
print("=" * 60)
