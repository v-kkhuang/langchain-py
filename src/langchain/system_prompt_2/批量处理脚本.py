# 安装依赖：pip install openai

import json
from openai import OpenAI

client = OpenAI()  # 需要设置 OPENAI_API_KEY 环境变量

# 1. 准备 Prompt 模板（就是你前面用的那段）
PROMPT = """你是一个新闻信息提取助手。只输出 JSON，不要其他文字。

规则：
1. 每个字段包含 value 和 confidence（0-1）
2. 未提及的字段 value 填 "未知"，confidence 填 0.1
3. 不要编造信息

--- 示例 1 ---
输入：3月12日，苹果公司在加州举办发布会，蒂姆·库克展示新款iPad，起售价799美元。
输出：
{{
  "事件名称": {{ "value": "苹果发布会", "confidence": 0.95 }},
  "时间": {{ "value": "3月12日", "confidence": 0.90 }},
  "地点": {{ "value": "加州", "confidence": 0.90 }},
  "人物": {{ "value": ["蒂姆·库克"], "confidence": 0.95 }},
  "组织": {{ "value": ["苹果公司"], "confidence": 0.98 }},
  "关键数字": {{ "value": ["799美元"], "confidence": 0.90 }}
}}

--- 示例 2 ---
输入：近日某企业公布新战略，据传投入数十亿。
输出：
{{
  "事件名称": {{ "value": "新战略公布", "confidence": 0.75 }},
  "时间": {{ "value": "未知", "confidence": 0.10 }},
  "地点": {{ "value": "未知", "confidence": 0.10 }},
  "人物": {{ "value": [], "confidence": 0.10 }},
  "组织": {{ "value": ["某企业"], "confidence": 0.60 }},
  "关键数字": {{ "value": ["数十亿"], "confidence": 0.65 }}
}}

--- 请提取以下新闻 ---
输入："""

# 2. 准备文档列表
documents = [
    {{"id": "doc_1", "text": "2024年6月20日，华为在深圳举行发布会..."}},
    {{"id": "doc_2", "text": "近日某企业公布新战略..."}},
    {{"id": "doc_3", "text": "3月15日，国家发改委在北京召开..."}},
]

# 3. 验证函数
def validate(raw):
    try:
        data = json.loads(raw)
    except:
        return None, "JSON解析失败"

    required = ["事件名称", "时间", "地点", "人物", "组织", "关键数字"]
    for f in required:
        if f not in data:
            return None, f"缺少字段 {f}"
    return data, "OK"

# 4. 批量处理
results = []
for doc in documents:
    print(f"正在处理 {doc['id']}...")

    # 调用 API
    resp = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {{"role": "user", "content": PROMPT + doc["text"]}}
        ],
        response_format={{"type": "json_object"}}  # 强制 JSON
    )
    raw = resp.choices[0].message.content

    # 验证
    data, msg = validate(raw)
    if data:
        data["_id"] = doc["id"]
        data["_status"] = "success"
        results.append(data)
        print(f"  ✓ 成功")
    else:
        results.append({{"_id": doc["id"], "_status": "error", "_error": msg}})
        print(f"  ✗ 失败: {msg}")

# 5. 保存结果
with open("results.json", "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

print(f"\n完成！成功 {sum(1 for r in results if r['_status']=='success')}/{len(results)} 篇")
print("结果已保存到 results.json")