"""
第一阶段 · 第四步拓展：高级提示技巧
=====================================
  1. Few-Shot PromptTemplate  —— 给模型几个「示例」，学会固定格式输出
  2. partial 部分绑定         —— 提前填一部分变量，复用模板
  3. Pipeline 组合            —— 两个模板串联（先总结，再翻译）

运行方式：
    uv run python examples/04_高级提示技巧.py
"""

import sys
from dotenv import load_dotenv
from langchain_core.prompts import (
    ChatPromptTemplate,
    FewShotChatMessagePromptTemplate,
    PromptTemplate,
)
from langchain_core.runnables import RunnablePassthrough
from langchain_deepseek import ChatDeepSeek

sys.stdout.reconfigure(encoding="utf-8")
load_dotenv()

llm = ChatDeepSeek(model="deepseek-chat", temperature=0)


# ─────────────────────────────────────────────────
# Demo 1: Few-Shot（少样本）—— 给示例让模型学格式
# ─────────────────────────────────────────────────
def demo_1_few_shot():
    print("=" * 60)
    print("🎯 Demo 1: Few-Shot 少样本提示（让模型学会固定格式）")
    print("=" * 60)

    # 1️⃣ 先准备若干「示例」，每个示例是 human 问 + ai 答
    examples = [
        {"input": "今天太高兴了！", "output": "emotion=positive, score=9, keywords=[高兴]"},
        {"input": "这个产品非常难用，差评。", "output": "emotion=negative, score=2, keywords=[难用,差评]"},
        {"input": "今天天气还可以，不冷不热。", "output": "emotion=neutral, score=5, keywords=[天气]"},
        {"input": "真的崩溃了，bug 修了三天还没好！", "output": "emotion=negative, score=1, keywords=[崩溃,bug]"},
    ]

    # 2️⃣ 单个示例的模板：一条 human + 一条 ai
    example_prompt = ChatPromptTemplate.from_messages([
        ("human", "{input}"),
        ("ai", "{output}"),
    ])

    # 3️⃣ 少样本模板：会自动把所有 examples 插入到 system 之后、用户输入之前
    few_shot_prompt = FewShotChatMessagePromptTemplate(
        example_prompt=example_prompt,
        examples=examples,
    )

    # 4️⃣ 拼到最终的 ChatPromptTemplate 里
    final_prompt = ChatPromptTemplate.from_messages([
        ("system", "你是一个情感分析器，严格按示例格式输出，不要解释。"),
        few_shot_prompt,  # ← 这里会展开成 4 组 human+ai 示例
        ("human", "{input}"),
    ])

    # 看看最终长什么样
    print("🔍 最终模板的 messages 结构：")
    preview = final_prompt.invoke({"input": "用户测试输入"})
    for i, msg in enumerate(preview.to_messages()):
        content_preview = msg.content[:40].replace("\n", " ")
        print(f"   {i}. [{msg.type.upper()}] {content_preview}...")
    print()

    # 5️⃣ 用管道跑真实请求
    chain = final_prompt | llm
    test_cases = [
        "这个电影太棒了，看了三遍还想再看！",
        "快递拖了一周才到，包装还破了。",
        "今天周三，明天周四。",
    ]
    for text in test_cases:
        result = chain.invoke({"input": text})
        print(f"   输入：{text}")
        print(f"   → 输出：{result.content}\n")

    print("💡 小结：Few-Shot 适合『有严格格式要求』的任务（分类、抽取、风格模仿）")
    print("   示例 3~5 个通常足够，示例越多样效果越好")


# ─────────────────────────────────────────────────
# Demo 2: partial 部分绑定 —— 预填一部分变量，减少重复
# ─────────────────────────────────────────────────
def demo_2_partial():
    print("\n" + "=" * 60)
    print("🧩 Demo 2: partial 部分绑定变量（减少重复传参）")
    print("=" * 60)

    # 模板有 4 个变量，但其实 role 和 style 对整个程序是固定的
    base_template = PromptTemplate.from_template(
        "你是{role}，请用{style}的风格写一段关于{topic}的话，字数控制在{word_count}字以内。"
    )

    # 场景：我有一个「幽默型程序员博主」的固定人设，每次只换 topic 和字数
    # 用 .partial() 先绑死 role 和 style
    blogger_template = base_template.partial(
        role="一个爱讲冷笑话的程序员博主",
        style="幽默吐槽+技术梗",
    )
    print(f"📋 绑定后剩余变量：{blogger_template.input_variables}")  # 只剩 topic, word_count

    # 之后调用只需要传剩下的两个变量
    chain_a = blogger_template | llm
    result_a = chain_a.invoke({"topic": "程序员加班", "word_count": "80"})
    print("📝 博主写『程序员加班』(80字)：")
    print(f"   {result_a.content}\n")

    # 同一个基础模板，换一套绑定 → 变成了「严肃的学术编辑」
    editor_template = base_template.partial(
        role="严谨的学术期刊编辑",
        style="正式、客观、精准",
    )
    chain_b = editor_template | llm
    result_b = chain_b.invoke({"topic": "程序员加班", "word_count": "80"})
    print("📝 学术编辑写『程序员加班』(80字)：")
    print(f"   {result_b.content}\n")

    print("💡 小结：partial 适合『程序里有固定人设/语言』的场景")
    print("   把不变的变量绑掉，调用时只传会变的，代码更干净")


# ─────────────────────────────────────────────────
# Demo 3: 双模板串联 Pipeline（先总结 → 再翻译）
# ─────────────────────────────────────────────────
def demo_3_pipeline():
    print("\n" + "=" * 60)
    print("🔗 Demo 3: 多模板串联（RunnablePassthrough 传数据）")
    print("=" * 60)

    # 场景：输入一段长中文 → 先用中文总结 → 再翻译成英文
    summarize_prompt = ChatPromptTemplate.from_template(
        "请用 3 个要点总结以下内容，每个要点不超过 20 字：\n\n{text}"
    )
    translate_prompt = ChatPromptTemplate.from_template(
        "把以下内容翻译成地道英文：\n\n{summarized}"
    )

    # 第一步单独的链：总结
    summarize_chain = summarize_prompt | llm

    # ✨ 关键：用 RunnablePassthrough 把『第一步的输出』传给第二步的输入
    #   assign() 会在原 dict 里加一个新字段 'summarized'
    from langchain_core.output_parsers import StrOutputParser
    parser = StrOutputParser()  # 把 AIMessage 转成 str，下一步才好拼接

    full_chain = (
        {"summarized": summarize_chain | parser, "text": RunnablePassthrough()}
        | translate_prompt
        | llm
    )

    long_text = """
    LangChain 是一个用于构建大语言模型应用的开源框架。
    它提供了提示模板、链式调用、智能体、记忆模块等核心组件。
    通过 LCEL 表达式语言，开发者可以用管道符声明式地组合复杂工作流。
    最新的 1.x 版本构建在 LangGraph 运行时之上，支持持久化、流式处理和人在回路。
    目前已被广泛用于 RAG 检索问答、智能客服、数据分析、自动化办公等场景。
    """
    print("📝 输入长文本（截取前 80 字）：")
    print(f"   {long_text.strip()[:80]}...\n")

    # 注意：因为我们用了 RunnablePassthrough()，这里 invoke 传一个 dict，
    # 'text' 键会被 passthrough 保留，'summarized' 键是第一步生成的
    result = full_chain.invoke({"text": long_text})
    print("🌐 最终输出（3 要点总结 + 英文翻译）：")
    print(f"   {result.content.replace(chr(10), chr(10) + '   ')}")

    print("\n💡 小结：这就是 LCEL 管道组合的雏形")
    print("   RunnablePassthrough 让上游数据在下游继续可见")
    print("   后面学 OutputParser 会把 StrOutputParser 讲清楚")


if __name__ == "__main__":
    print("🚀 第一阶段 · 第四步拓展：高级提示技巧\n")

    demo_1_few_shot()
    demo_2_partial()
    demo_3_pipeline()

    print("\n🎉 高级提示技巧完成！下一步：输出解析器 OutputParser")
