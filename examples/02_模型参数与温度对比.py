"""
第一阶段 · 第三步拓展：理解模型参数
=====================================
重点对比：
  1. temperature —— 控制「创造性」，0=最确定，2=最放飞
  2. max_tokens  —— 限制输出长度
  3. 多轮对话消息格式 —— system / human / ai 的正确使用

运行方式：
    uv run python examples/02_模型参数与温度对比.py
"""

import sys
from dotenv import load_dotenv
from langchain_deepseek import ChatDeepSeek

sys.stdout.reconfigure(encoding="utf-8")
load_dotenv()

QUESTION = "给一个创意产品起 3 个名字，产品是：能自动浇花的智能花盆"


def compare_temperature():
    """对比 temperature 从 0 → 2 的输出差异"""
    print("=" * 60)
    print("🌡️  Demo 1: temperature 对比（问同一个问题 3 次）")
    print("=" * 60)
    print(f"❓ 问题：{QUESTION}\n")

    temps = [0, 0.7, 1.5]
    for t in temps:
        llm = ChatDeepSeek(model="deepseek-chat", temperature=t)
        print(f"── temperature={t} ──")
        result = llm.invoke(QUESTION)
        # 处理换行，让输出紧凑
        lines = [f"    {ln}" for ln in result.content.strip().splitlines()]
        print("\n".join(lines))
        print()

    print("💡 小结：")
    print("   temperature=0   → 每次答案几乎一样，适合分类、抽取、SQL 生成")
    print("   temperature=0.7 → 平衡，日常问答默认值")
    print("   temperature=1.5+ → 每次都很放飞，适合起名、写诗、创意写作")
    print("   ⚠️  值太高会出现胡言乱语，要谨慎")


def demo_max_tokens():
    """演示 max_tokens 截断输出"""
    print("\n" + "=" * 60)
    print("✂️  Demo 2: max_tokens 限制输出长度")
    print("=" * 60)

    llm_short = ChatDeepSeek(model="deepseek-chat", temperature=0, max_tokens=50)
    llm_long = ChatDeepSeek(model="deepseek-chat", temperature=0, max_tokens=300)

    question = "详细介绍 LangChain 的作用"
    print(f"❓ 问题：{question}\n")

    r_short = llm_short.invoke(question)
    print(f"── max_tokens=50 ──")
    print(f"   {r_short.content[:100]}...")
    print(f"   (已被截断，len={len(r_short.content)} 字符)\n")

    r_long = llm_long.invoke(question)
    print(f"── max_tokens=300 ──")
    print(f"   {r_long.content[:200]}...")
    print(f"   len={len(r_long.content)} 字符\n")

    print("💡 小结：")
    print("   max_tokens 限制『最多生成多少个 token』")
    print("   1 token ≈ 1.5~2 个中文字符，不是精准对应")
    print("   场景：短回复用 50~200，长文总结用 500~2000")


def demo_multi_turn():
    """演示多轮对话：正确拼接历史消息"""
    print("\n" + "=" * 60)
    print("💬 Demo 3: 多轮对话消息格式")
    print("=" * 60)

    llm = ChatDeepSeek(model="deepseek-chat", temperature=0)

    # ❌ 错误示范：只发最新问题，模型「失忆」
    print("❌ 错误示范（模型失忆）：")
    wrong_messages = [
        {"role": "human", "content": "我叫小明，今年 10 岁"},  # 这句说了但没保留
        {"role": "human", "content": "我今年多大了？"},           # 只发这句
    ]
    result = llm.invoke(wrong_messages)
    print(f"   问：我今年多大了？")
    print(f"   答：{result.content[:80]}")
    print("   👉 模型不知道，因为没带上一轮历史\n")

    # ✅ 正确示范：带上完整历史（system + human + ai + human ...）
    print("✅ 正确示范（带上历史）：")
    correct_messages = [
        {"role": "system", "content": "你是小明的好朋友，说话要亲切。"},
        {"role": "human", "content": "我叫小明，今年 10 岁"},
        {"role": "ai", "content": "你好小明！很高兴认识你，10 岁正是充满好奇心的年纪呀～"},
        {"role": "human", "content": "我今年多大了？"},
    ]
    result = llm.invoke(correct_messages)
    print(f"   问：我今年多大了？")
    print(f"   答：{result.content}")
    print("   👉 模型能正确回答，因为消息数组里包含了完整上下文\n")

    print("💡 小结：")
    print("   消息格式严格按 role 区分：")
    print("     system  → 人设/规则，只发一次，在最前面")
    print("     human   → 用户说的话")
    print("     ai      → 模型的回复")
    print("   多轮对话 = 把上面三类按时间顺序 append 成一个数组")
    print("   ⚠️  不能只保留最新一句，否则模型没有上下文")


if __name__ == "__main__":
    print("🚀 第一阶段 · 模型参数深度理解\n")

    compare_temperature()
    demo_max_tokens()
    demo_multi_turn()

    print("\n🎉 参数理解完毕！下一步：提示模板 PromptTemplate")
