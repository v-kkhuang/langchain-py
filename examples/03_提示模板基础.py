"""
第一阶段 · 第四步：提示模板基础
=================================
学习三种核心模板：
  1. PromptTemplate            —— 最简单的字符串模板（单轮场景）
  2. ChatPromptTemplate        —— 聊天场景模板（system/human/ai 多角色）
  3. from_messages 消息模板    —— 最灵活，支持消息占位符

核心概念：
  - 模板里用 {变量名} 做占位符
  - .format() / .invoke() 传入变量，得到可发送给模型的 Prompt
  - 模板本身也是 Runnable，可以直接用 | 拼到 llm 前面（即 LCEL）

运行方式：
    uv run python examples/03_提示模板基础.py
"""

import sys
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_deepseek import ChatDeepSeek

sys.stdout.reconfigure(encoding="utf-8")
load_dotenv()

llm = ChatDeepSeek(model="deepseek-chat", temperature=0)


# ─────────────────────────────────────────────────
# Demo 1: PromptTemplate — 字符串级模板
# ─────────────────────────────────────────────────
def demo_1_simple_prompt():
    print("=" * 60)
    print("📌 Demo 1: PromptTemplate 基础（字符串模板）")
    print("=" * 60)

    # ① 定义模板，{topic} 是占位符变量
    template = "请用3句话解释什么是{topic}，语言要通俗易懂。"
    prompt = PromptTemplate.from_template(template)

    # ② 查看模板需要哪些变量
    print(f"📋 模板变量：{prompt.input_variables}")  # ['topic']

    # ③ 用 .format() 填充变量 → 得到纯字符串
    filled = prompt.format(topic="大语言模型")
    print(f"📝 填充后：{filled}\n")

    # ④ 实际调用模型（方式 A：先 format，再 invoke）
    result = llm.invoke(filled)
    print("💡 回答：")
    print(f"   {result.content.replace(chr(10), chr(10) + '   ')}\n")

    # ⑤ 方式 B：模板是 Runnable，直接和 llm 用 | 拼接（LCEL 预演）
    #    prompt | llm 是最常见的组合，后面第五步 OutputParser 会再加一环
    chain = prompt | llm
    result2 = chain.invoke({"topic": "区块链"})
    print("🔗 用 LCEL 管道 prompt|llm 问『区块链』：")
    print(f"   {result2.content.replace(chr(10), chr(10) + '   ')}")


# ─────────────────────────────────────────────────
# Demo 2: PromptTemplate — 多变量
# ─────────────────────────────────────────────────
def demo_2_multi_var():
    print("\n" + "=" * 60)
    print("📌 Demo 2: PromptTemplate 多变量")
    print("=" * 60)

    template = """你是一个{role}。
请为一位{audience}设计一个{duration}的{topic}学习计划，
要求：
1. 每天的任务要具体可执行
2. 包含理论和实践两部分
3. 整体风格{style}"""

    prompt = PromptTemplate.from_template(template)
    print(f"📋 模板变量：{prompt.input_variables}")

    # 一次填 5 个变量
    filled = prompt.format(
        role="编程教育专家",
        audience="完全零基础的大学生",
        duration="7天",
        topic="Python 入门",
        style="轻松幽默，像和朋友聊天一样",
    )
    print("📝 填充后的完整提示词（截取前 200 字）：")
    print(f"   {filled[:200]}...\n")

    # 直接走管道
    chain = prompt | llm
    result = chain.invoke({
        "role": "健身教练",
        "audience": "久坐的程序员",
        "duration": "30天",
        "topic": "改善颈椎和腰椎",
        style="简洁直接，用数字列出动作",
    })
    print("💡 回答（30天颈椎康复计划）：")
    print(f"   {result.content.replace(chr(10), chr(10) + '   ')}")


# ─────────────────────────────────────────────────
# Demo 3: ChatPromptTemplate — 聊天场景模板（重点）
# ─────────────────────────────────────────────────
def demo_3_chat_prompt():
    print("\n" + "=" * 60)
    print("📌 Demo 3: ChatPromptTemplate 聊天模板（⭐重点）")
    print("=" * 60)

    # 聊天模板是「消息列表」，每条消息有 role 和 内容模板
    chat_prompt = ChatPromptTemplate.from_messages([
        # 第 1 条：system 角色，固定人设（可以含变量）
        ("system", "你是一个{personality}的{language}老师，回答全程用{language}。"),
        # 第 2 条：human 角色，用户问题（含变量）
        ("human", "帮我把这句话翻译成{language}：{sentence}"),
    ])

    print(f"📋 模板变量：{chat_prompt.input_variables}")

    # 填充后得到的是「ChatPromptValue」，可以直接喂给 llm
    filled = chat_prompt.invoke({
        "personality": "活泼爱讲笑话",
        "language": "英语",
        "sentence": "今天的天气适合在家写代码。",
    })
    print("📝 填充后的消息列表：")
    for msg in filled.to_messages():
        print(f"   [{msg.type.upper()}] {msg.content}")
    print()

    # 管道拼接
    chain = chat_prompt | llm
    result = chain.invoke({
        "personality": "严格又专业",
        "language": "日语",
        "sentence": "我想学 LangChain，请给我推荐入门资源。",
    })
    print(f"💡 日语老师的回答：{result.content}")


# ─────────────────────────────────────────────────
# Demo 4: 占位符 {chat_history} — 为多轮对话留位置
# ─────────────────────────────────────────────────
def demo_4_history_placeholder():
    print("\n" + "=" * 60)
    print("📌 Demo 4: MessagesPlaceholder 多轮对话占位符")
    print("=" * 60)

    from langchain_core.prompts import MessagesPlaceholder

    chat_prompt = ChatPromptTemplate.from_messages([
        ("system", "你是一个记忆大师，能记住用户说过的所有信息。"),
        # 🎯 用 MessagesPlaceholder，给对话历史留一个动态插入的位置
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
    ])
    print(f"📋 模板变量：{chat_prompt.input_variables}")

    # chat_history 传「消息列表」，input 传用户最新问题
    from langchain_core.messages import HumanMessage, AIMessage
    chain = chat_prompt | llm
    result = chain.invoke({
        "chat_history": [
            HumanMessage(content="我最喜欢的颜色是天蓝色"),
            AIMessage(content="记下来了！你喜欢天蓝色～"),
            HumanMessage(content="我家住在杭州"),
            AIMessage(content="好的，杭州是座美丽的城市。"),
        ],
        "input": "结合我告诉你的信息，给我写一句自我介绍。",
    })
    print("💡 回答（模型用了 chat_history 里的信息）：")
    print(f"   {result.content}")


if __name__ == "__main__":
    print("🚀 第一阶段 · 第四步：提示模板基础\n")

    demo_1_simple_prompt()
    demo_2_multi_var()
    demo_3_chat_prompt()
    demo_4_history_placeholder()

    print("\n🎉 提示模板基础完成！下一步：高级技巧（少样本、部分绑定）")
