import json
import os
import sys

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

sys.stdout.reconfigure(encoding="utf-8")

# 锚定到脚本所在目录，避免从其他工作目录运行时找不到文件
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

load_dotenv()


llm = ChatOpenAI(
    model=os.getenv("DEEPSEEK_MODEL"),
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url=os.getenv("DEEPSEEK_API_URL"),
)


def load_faq(filepath: str = None) -> list:
    if filepath is None:
        filepath = os.path.join(BASE_DIR, "faq_knowledge_base.json")
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def search_faq_keyword(query: str, faq_list: list) -> list:
    """关键词匹配：返回与用户问题最相关的 FAQ 条目。"""
    scored = []
    for item in faq_list:
        score = 0
        for kw in item.get("keywords", []):
            if kw.lower() in query.lower():
                score += 1
        # 同时检查问题文本本身的重叠
        query_words = set(query.lower().split())
        question_words = set(item["question"].lower().split())
        overlap = len(query_words & question_words)
        score += overlap * 0.5

        if score > 0:
            scored.append((score, item))

    # 按分数降序排列
    scored.sort(key=lambda x: x[0], reverse=True)
    return [item for _, item in scored[:3]]  # 返回 top 3


def build_context_prompt(query: str, matched_faqs: list) -> str:
    """把匹配到的 FAQ 注入到 prompt 中。"""
    if not matched_faqs:
        return ""  # 没有匹配到，留给 Step 4 处理

    context_text = "以下是知识库中的相关信息：\n\n"
    for i, faq in enumerate(matched_faqs, 1):
        context_text += f"[参考{i}] (ID: {faq['id']})\n"
        context_text += f"问题：{faq['question']}\n"
        context_text += f"回答：{faq['answer']}\n\n"

    return context_text


def answer_question(query: str) -> str:
    faq_list = load_faq()
    matched = search_faq_keyword(query, faq_list)

    if not matched:
        return "抱歉，我没有找到与您问题相关的信息。"

    context = build_context_prompt(query, matched)

    system_prompt = "你是一位客服助手。请根据提供的知识库信息回答用户问题。"
    user_prompt = f"{context}\n用户问题：{query}"

    response = llm.invoke(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
    )
    return response.content


# 测试
print(answer_question("你们告诉我怎么全部退掉"))
print("---")
print(answer_question("忘记密码了怎么办？"))
