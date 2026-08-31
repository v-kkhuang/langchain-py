"""
第一阶段 · 第三步：模型调用基础
=================================
覆盖 Runnable 接口的 4 种核心调用方式：
1. invoke    同步单次调用
2. stream    同步流式输出（逐字打印）
3. batch     同步批量调用
4. ainvoke   异步单次调用（适合 Web 服务场景）

运行方式：
    uv run python examples/01_基础模型调用.py
"""

import asyncio
import sys

import time
from dotenv import load_dotenv
from langchain_deepseek import ChatDeepSeek

# Windows 终端默认 GBK，强制 UTF-8 避免 emoji/中文乱码
sys.stdout.reconfigure(encoding="utf-8")

load_dotenv()  # 从 .env 读取 DEEPSEEK_API_KEY

# ──────────────────────────────────────────────
# 初始化模型
# ──────────────────────────────────────────────
# LangChain 1.x 推荐两种方式：
#   A) provider 专属包：ChatDeepSeek（本示例）
#   B) 统一工厂：init_chat_model("deepseek:deepseek-chat")
# temperature=0 表示"最确定性"，回答更稳定，适合学习调试
llm = ChatDeepSeek(model="deepseek-chat", temperature=0)


def demo_1_invoke():
    """【方式1】invoke — 同步单次调用，一次性返回完整结果"""
    print("\n" + "=" * 60)
    print("📌 Demo 1: invoke 同步单次调用")
    print("=" * 60)

    # messages 格式：角色 (system/human/ai) + 内容
    messages = [
        {"role": "system", "content": "你是一位简洁的编程老师，回答不超过 3 句话。"},
        {"role": "human", "content": "用一句话解释什么是 LangChain？"},
    ]

    result = llm.invoke(messages)
    # 返回的是 AIMessage 对象，.content 取出文本
    print(f"✅ 完整回答：{result.content}")
    print(f"🔍 回答类型：{type(result).__name__}")
    print(f"🔍 元数据（token 用量等）：{result.response_metadata.get('token_usage', 'N/A')}")


def demo_2_stream():
    """【方式2】stream — 流式输出，逐 token 返回（体验更好）"""
    print("\n" + "=" * 60)
    print("📌 Demo 2: stream 流式输出（逐字打印）")
    print("=" * 60)

    messages = [
        {"role": "system", "content": "你是一位科普作家，用生动的语言解释。"},
        {"role": "human", "content": "解释什么是大语言模型的 Embedding？"},
    ]

    print("💬 回答：", end="", flush=True)
    collected = ""
    for chunk in llm.stream(messages):
        # 每个 chunk 是一小段 AIMessageChunk
        collected += chunk.content
        print(chunk.content, end="", flush=True)
    print()  # 换行
    print(f"🔍 共收到 {len(collected)} 个字符")


def demo_3_batch():
    """【方式3】batch — 批量调用，多个问题并行处理，返回列表"""
    print("\n" + "=" * 60)
    print("📌 Demo 3: batch 批量调用（3 个问题并行）")
    print("=" * 60)

    # 每个元素是一组独立的 messages
    batch_inputs = [
        [{"role": "human", "content": "Python 和 JavaScript 哪个更适合 AI 开发？"}],
        [{"role": "human", "content": "解释什么是 API？"}],
        [{"role": "human", "content": "推荐 3 个学习编程的好习惯"}],
    ]

    start = time.time()
    results = llm.batch(batch_inputs, config={"max_concurrency": 3})
    elapsed = time.time() - start

    for i, res in enumerate(results, 1):
        # 截取前 50 字展示
        preview = res.content[:50].replace("\n", " ") + ("..." if len(res.content) > 50 else "")
        print(f"  Q{i}: {preview}")
    print(f"⏱️  总耗时：{elapsed:.2f} 秒（并行比串行快）")


async def demo_4_ainvoke():
    """【方式4】ainvoke — 异步调用（FastAPI 等 Web 场景常用）"""
    print("\n" + "=" * 60)
    print("📌 Demo 4: ainvoke 异步调用")
    print("=" * 60)

    messages = [
        {"role": "system", "content": "你是一个翻译官，只输出翻译结果，不要解释。"},
        {"role": "human", "content": "把 'LangChain 是 LLM 应用开发的瑞士军刀' 翻译成英文"},
    ]

    result = await llm.ainvoke(messages)
    print(f"🌐 翻译结果：{result.content}")


# ──────────────────────────────────────────────
# 入口：按顺序跑 4 个 Demo
# ──────────────────────────────────────────────
if __name__ == "__main__":
    print("🚀 第一阶段 · 模型调用基础练习")
    print(f"   使用模型：{llm.model}   temperature={llm.temperature}")

    demo_1_invoke()
    demo_2_stream()
    demo_3_batch()
    asyncio.run(demo_4_ainvoke())

    print("\n" + "🎉 四个调用方式演示完毕，下一站：提示模板！")
