"""
Chroma 向量数据库入门示例
========================
本示例演示了 Chroma 的基本用法：
1. 创建/获取集合 (Collection)
2. 添加文档和向量
3. 相似度搜索
4. 带过滤条件的搜索
5. 更新和删除数据
6. 持久化存储

注意：为避免下载 79MB 的 ONNX 模型，本示例使用基于词袋的简单嵌入函数。
实际项目中可使用 sentence-transformers / OpenAI 等嵌入模型。
"""

import chromadb
import os
import shutil
import hashlib
import re
import math
from collections import Counter

# ============================================================
# 简单嵌入函数：基于词袋 + TF-IDF 思想的本地向量化
# 维度: 128，通过词哈希映射到固定维度
# ============================================================
EMBED_DIM = 128


def simple_embed(text: str, dim: int = EMBED_DIM) -> list:
    """将文本转换为固定维度的向量（基于词袋+哈希）"""
    # 中文按字分词，英文按词分词
    words = re.findall(r'[\u4e00-\u9fff]|[a-zA-Z]+', text.lower())
    if not words:
        return [0.0] * dim

    vec = [0.0] * dim
    word_counts = Counter(words)
    total = sum(word_counts.values())

    for word, count in word_counts.items():
        # 用词的哈希值确定向量维度索引
        h = int(hashlib.md5(word.encode()).hexdigest(), 16)
        idx = h % dim
        # TF 权重
        tf = count / total
        vec[idx] += tf

    # L2 归一化
    norm = math.sqrt(sum(v * v for v in vec))
    if norm > 0:
        vec = [v / norm for v in vec]
    return vec


def embed_batch(texts: list) -> list:
    """批量生成向量"""
    return [simple_embed(t) for t in texts]


# ============================================================
# 示例 1: 基础用法 —— 内存模式
# ============================================================
def demo_basic():
    print("=" * 60)
    print("【示例 1】基础用法 —— 内存模式 + 自定义嵌入")
    print("=" * 60)

    client = chromadb.Client()

    collection_name = "book_collection"
    try:
        client.delete_collection(collection_name)
    except Exception:
        pass

    collection = client.create_collection(
        name=collection_name,
        metadata={"description": "图书推荐系统示例"}
    )

    documents = [
        "三体 是 刘慈欣 创作 的 科幻 小说 讲述 地球 文明 与 三体 文明 接触 对抗",
        "活着 是 余华 的 代表作 描写 福贵 一生 坎坷 经历 展现 生命 韧性",
        "百年孤独 是 马尔克斯 的 魔幻 现实主义 经典 讲述 布恩迪亚 家族 七代 故事",
        "红楼梦 是 中国 古典 四大名著 贾宝玉 林黛玉 爱情 悲剧 主线",
        "人类简史 从 认知 革命 讲到 科学 革命 重新 审视 人类 发展 历程",
        "深度学习 是 Ian Goodfellow 经典 教材 系统 介绍 深度学习 理论",
        "小王子 是 安托万 圣埃克苏佩里 童话 讲述 小王子 星际 旅行 见闻",
        "明朝那些事儿 幽默 风趣 笔触 讲述 明朝 三百年 历史",
    ]

    metadatas = [
        {"title": "三体", "author": "刘慈欣", "category": "科幻", "year": 2008},
        {"title": "活着", "author": "余华", "category": "文学", "year": 1993},
        {"title": "百年孤独", "author": "马尔克斯", "category": "文学", "year": 1967},
        {"title": "红楼梦", "author": "曹雪芹", "category": "古典文学", "year": 1791},
        {"title": "人类简史", "author": "尤瓦尔·赫拉利", "category": "历史", "year": 2011},
        {"title": "深度学习", "author": "Ian Goodfellow", "category": "科技", "year": 2016},
        {"title": "小王子", "author": "圣-埃克苏佩里", "category": "童话", "year": 1943},
        {"title": "明朝那些事儿", "author": "当年明月", "category": "历史", "year": 2006},
    ]

    ids = [f"book_{i}" for i in range(len(documents))]

    # 手动生成向量并添加
    embeddings = embed_batch(documents)
    collection.add(
        documents=documents,
        embeddings=embeddings,
        metadatas=metadatas,
        ids=ids
    )

    print(f"  已添加 {collection.count()} 条文档到集合")

    # 相似度搜索
    query = "深度学习 人工智能 科学 技术"
    query_emb = simple_embed(query)
    print(f"\n  搜索: '{query}'")
    results = collection.query(
        query_embeddings=[query_emb],
        n_results=3
    )

    print("\n  搜索结果 (按相似度排序):")
    for i, (doc, meta, dist) in enumerate(zip(
        results['documents'][0],
        results['metadatas'][0],
        results['distances'][0]
    ), 1):
        sim = 1 - dist
        print(f"  {i}. 《{meta['title']}》 - {meta['author']} | 类别: {meta['category']} | 相似度: {sim:.4f}")

    return collection


# ============================================================
# 示例 2: 带元数据过滤的搜索
# ============================================================
def demo_with_filters(collection):
    print("\n" + "=" * 60)
    print("【示例 2】带元数据过滤的搜索")
    print("=" * 60)

    # 只在「历史」类书籍中搜索
    query = "明朝 历史 三百年"
    query_emb = simple_embed(query)
    print(f"\n  在「历史」类中搜索: '{query}'")
    results = collection.query(
        query_embeddings=[query_emb],
        n_results=3,
        where={"category": "历史"}
    )

    print("  搜索结果:")
    for i, (doc, meta, dist) in enumerate(zip(
        results['documents'][0],
        results['metadatas'][0],
        results['distances'][0]
    ), 1):
        print(f"  {i}. 《{meta['title']}》 - 相似度: {1 - dist:.4f}")

    # 按年份范围过滤
    query2 = "人类 发展 历程"
    query_emb2 = simple_embed(query2)
    print(f"\n  搜索 2000 年后出版的书中关于「人类发展」的内容")
    results = collection.query(
        query_embeddings=[query_emb2],
        n_results=3,
        where={"year": {"$gte": 2000}}
    )

    print("  搜索结果:")
    for i, (doc, meta, dist) in enumerate(zip(
        results['documents'][0],
        results['metadatas'][0],
        results['distances'][0]
    ), 1):
        print(f"  {i}. 《{meta['title']}》 ({meta['year']}年) - 相似度: {1 - dist:.4f}")


# ============================================================
# 示例 3: 持久化存储
# ============================================================
def demo_persistent():
    print("\n" + "=" * 60)
    print("【示例 3】持久化存储 —— 数据保存到本地磁盘")
    print("=" * 60)

    persist_dir = "./chroma_data"

    if os.path.exists(persist_dir):
        shutil.rmtree(persist_dir)

    client = chromadb.PersistentClient(path=persist_dir)
    collection = client.create_collection(name="notes")

    docs = [
        "向量数据库 可以 用来 做 语义搜索",
        "Chroma 轻量级 向量数据库 非常 适合 入门学习",
        "嵌入模型 可以 把 文本 转换 高维向量 语义 相近 文本 向量 距离 近",
    ]
    embeddings = embed_batch(docs)

    collection.add(
        documents=docs,
        embeddings=embeddings,
        metadatas=[
            {"topic": "学习笔记", "date": "2026-09-10"},
            {"topic": "技术笔记", "date": "2026-09-10"},
            {"topic": "概念理解", "date": "2026-09-11"},
        ],
        ids=["note_1", "note_2", "note_3"]
    )

    print(f"  数据已保存到: {os.path.abspath(persist_dir)}")
    print(f"  集合中有 {collection.count()} 条笔记")

    # 重新加载验证持久化
    client2 = chromadb.PersistentClient(path=persist_dir)
    collection2 = client2.get_collection("notes")
    print(f"  重新加载后仍有 {collection2.count()} 条笔记，持久化成功!")

    # 测试搜索
    query = "什么是 语义搜索 向量"
    query_emb = simple_embed(query)
    results = collection2.query(
        query_embeddings=[query_emb],
        n_results=2
    )
    print(f"\n  搜索 '{query}' 的最相关笔记:")
    for doc, dist in zip(results['documents'][0], results['distances'][0]):
        print(f"    相似度 {1 - dist:.4f}: {doc[:50]}...")


# ============================================================
# 示例 4: 增删改查 (CRUD)
# ============================================================
def demo_crud():
    print("\n" + "=" * 60)
    print("【示例 4】增删改查 (CRUD) 操作")
    print("=" * 60)

    client = chromadb.Client()
    collection = client.create_collection("crud_demo")

    docs = ["文档A 内容", "文档B 内容", "文档C 内容"]
    embs = embed_batch(docs)

    # Create
    collection.add(
        documents=docs,
        embeddings=embs,
        ids=["a", "b", "c"]
    )
    print(f"  添加后: {collection.count()} 条")

    # Read
    result = collection.get(ids=["a", "b"])
    print(f"  查询 id=a,b: {result['documents']}")

    # Update
    new_emb = simple_embed("文档A 已更新 内容")
    collection.update(
        ids=["a"],
        documents=["文档A（已更新）"],
        embeddings=[new_emb]
    )
    result = collection.get(ids=["a"])
    print(f"  更新后 id=a: {result['documents'][0]}")

    # Delete
    collection.delete(ids=["b"])
    remaining = collection.get()
    print(f"  删除 id=b 后剩余: {collection.count()} 条 (剩余: {remaining['ids']})")


# ============================================================
# 主函数
# ============================================================
def main():
    print("\n" + "=" * 60)
    print("       Chroma 向量数据库入门示例")
    print("=" * 60 + "\n")

    collection = demo_basic()
    demo_with_filters(collection)
    demo_persistent()
    demo_crud()

    print("\n" + "=" * 60)
    print("  所有示例运行完成!")
    print("=" * 60)
    print("\n  下一步建议:")
    print("  1. 修改示例中的文档，尝试搜索不同的内容")
    print("  2. 安装 sentence-transformers 使用真正的语义嵌入模型")
    print("     pip install sentence-transformers")
    print("  3. 结合 LangChain / LlamaIndex 构建 RAG 应用")
    print("  4. 探索 Chroma 服务端模式: chroma run --port 8000")


if __name__ == "__main__":
    main()
