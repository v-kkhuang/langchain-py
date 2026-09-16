from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
import os

load_dotenv()
llm = ChatOpenAI(
    model=os.environ["DEEPSEEK_MODEL"],  # 真实模型名
    base_url=os.environ["DEEPSEEK_API_URL"],
    api_key=os.environ["DEEPSEEK_API_KEY"],  # 手动传 key（因为变量名不是 OPENAI_API_KEY）
)


def summarize_with_role(text: str, role: str = "editor") -> str:
    """带角色提示的摘要。"""
    role_map = {
        "editor": "你是一位资深新闻编辑，擅长将冗长的文章提炼为核心信息。",
        "academic": "你是一位学术综述作者，擅长用严谨的语言概括研究内容。",
        "teacher": "你是一位擅长深入浅出的科普作者，能把复杂内容讲给普通人听。",
    }

    # 重点是这里，增加了系统提示词， role_map["editor"] 是role 参数没有key可以匹配时，默认的get，类似map.getordefalut
    system_prompt = role_map.get(role, role_map["editor"])
    user_prompt = f"请总结以下文本：\n\n{text}"
    response = llm.invoke(
        [
            {"role": "user", "content": user_prompt},
            {"role": "system", "content": system_prompt},
        ]
    )
    return response.content


sample_text = """
人工智能（AI）是计算机科学的一个分支，致力于创造能够模拟人类智能行为的系统。
近年来，深度学习技术的突破使得AI在图像识别、自然语言处理、语音识别等领域取得了显著进展。
然而，AI的发展也带来了伦理挑战，包括隐私保护、算法偏见、就业影响等问题。
专家认为，未来AI的发展需要在技术创新和伦理约束之间找到平衡。
"""

demo_text = """
今天妈妈抱我上楼的时候我碰到一个老阿姨，她不停的扮着鬼脸逗我笑，可是我一点也不觉得好玩，她脸上的每一个器官都有我的2倍大，我看到那硕大的眼睛凑近我的脸，不停的放大，缩小，恐惧感油然而生，我“哇”的一声哭了出来，老阿姨真害怕，但愿我以后别再碰到她
"""


for role in ["editor", "academic", "teacher"]:
    print(f"\n=== 角色: {role} ===")
    print(summarize_with_role(demo_text, role))


# === 角色: editor ===
# 人工智能在深度学习推动下取得显著进展，但伴随隐私、算法偏见、就业冲击等伦理挑战，未来发展需在技术创新与伦理约束间寻求平衡。

# === 角色: academic ===
# 人工智能（AI）是计算机科学中致力于模拟人类智能行为的重要分支。近年来，深度学习技术的突破显著推动了其在图像识别、自然语言处理和语音识别等领域的应用进步。然而，AI的快速发展同时引发了隐私保护、算法偏见及就业影响等伦理挑战。学界普遍认为，未来AI的发展应强调技术创新与伦理治理之间的动态平衡。

# === 角色: teacher ===
# 人工智能是一门让电脑像人一样思考的技术。这几年，深度学习让AI在识图、听懂人话、语音识别方面进步飞快。不过，它也带来了隐私泄露、算法偏见、取代工作等麻烦。专家说，未来得在技术创新和规矩约束之间找到平衡。


# demo_text返回
# === 角色: editor ===
# 核心信息：妈妈抱我上楼时遇到一位老阿姨，她扮鬼脸逗我，但我因她五官硕大感到恐惧而大哭，希望以后别再遇到她。

# === 角色: academic ===
# 这段文字以儿童第一人称视角，记述了被母亲抱上楼途中偶遇一位老年女性，对方持续以夸张面部表情逗弄“我”，但其五官尺寸远超“我”的认知范围，近距离放大缩小的面孔引发了“我”强烈的恐惧情绪，致使“我”当场大哭，并表达了日后不愿再遇的愿望。

# === 角色: teacher ===
# 这段文字以婴儿视角描述：被妈妈抱上楼时，遇到一位老阿姨不停扮鬼脸逗“我”，但她五官过于夸张（比“我”大两倍），凑近时让“我”感到恐惧，因此大哭，并希望以后别再遇到她。
