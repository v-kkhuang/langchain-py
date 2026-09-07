import argparse
import json
import os
import sys

from dotenv import load_dotenv
from glossary import Glossary
from langchain_openai import ChatOpenAI
from openai import OpenAI
from prompts import REWRITE_CONFIG, STYLE_CONFIG
from scoring import score_translation

sys.stdout.reconfigure(encoding="utf-8")

load_dotenv()


llm = ChatOpenAI(
    model=os.getenv("DEEPSEEK_MODEL"),
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url=os.getenv("DEEPSEEK_API_URL"),
)


def do_translate(text, source_lang, target_lang, style, glossary_path=None, iterative=False):
    """翻译流程。"""
    glossary = Glossary()
    if glossary_path:
        glossary.load_from_file(glossary_path)

    config = STYLE_CONFIG[style]
    system_prompt = f"{config['role']}\n源语言：{source_lang}\n目标语言：{target_lang}"
    glossary_text = glossary.format_for_prompt()

    user_prompt = config["instruction"]
    if glossary_text:
        user_prompt += f"\n\n{glossary_text}"
    user_prompt += f"\n\n请翻译：\n{text}"

    feedback = ""
    max_rounds = 3 if iterative else 1

    for i in range(max_rounds):
        current_prompt = user_prompt
        if feedback:
            current_prompt += f"\n\n【改进建议】\n{feedback}"

        response = llm.invoke(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]
        )
        translated = response.content
        scores = score_translation(text, translated, source_lang, target_lang, style)
        avg = sum([scores[k] for k in ["fluency", "accuracy", "style_match"]]) / 3

        if avg >= 8.0 or not iterative:
            break
        feedback = scores["reason"]

    return {"result": translated, "scores": scores, "rounds": i + 1}


def do_rewrite(text, mode):
    """改写流程。"""
    config = REWRITE_CONFIG[mode]
    response = llm.invoke(
        [
            {"role": "system", "content": config["role"]},
            {"role": "user", "content": f"{config['instruction']}\n\n原文：\n{text}"},
        ]
    )
    return response.content


def main():
    parser = argparse.ArgumentParser(description="多语言翻译与文本改写工具")
    sub = parser.add_subparsers(dest="command")

    # 翻译子命令
    t = sub.add_parser("translate", help="翻译文本")
    t.add_argument("input", help="输入文件路径")
    t.add_argument("--source", default="中文")
    t.add_argument("--target", default="English")
    t.add_argument("--style", choices=["literal", "free", "literary", "business"], default="free")
    t.add_argument("--glossary", help="术语表 JSON 文件路径")
    t.add_argument("--iterative", action="store_true")

    # 改写子命令
    r = sub.add_parser("rewrite", help="改写文本")
    r.add_argument("input", help="输入文件路径")
    r.add_argument(
        "--mode", choices=["formalize", "casualize", "simplify", "expand"], default="formalize"
    )

    args = parser.parse_args()

    with open(args.input, "r", encoding="utf-8") as f:
        text = f.read()

    if args.command == "translate":
        result = do_translate(
            text, args.source, args.target, args.style, args.glossary, args.iterative
        )
        print(f"\n{'=' * 50}")
        print(f"译文：\n{result['result']}")
        print(
            f"\n评分：流畅度 {result['scores']['fluency']} | 准确度 {result['scores']['accuracy']} | 风格匹配 {result['scores']['style_match']}"
        )
        print(f"迭代轮数：{result['rounds']}")
    elif args.command == "rewrite":
        result = do_rewrite(text, args.mode)
        print(f"\n{'=' * 50}")
        print(f"改写结果：\n{result}")


if __name__ == "__main__":
    main()
