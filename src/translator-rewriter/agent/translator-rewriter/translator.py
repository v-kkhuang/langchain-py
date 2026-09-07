import argparse
import json
import logging
import os
import sys

from dotenv import load_dotenv
from glossary import Glossary
from langchain_openai import ChatOpenAI
from openai import OpenAI
from prompts import REWRITE_CONFIG, STYLE_CONFIG
from scoring import score_translation

sys.stdout.reconfigure(encoding="utf-8")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("translator")

load_dotenv()


llm = ChatOpenAI(
    model=os.getenv("DEEPSEEK_MODEL"),
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url=os.getenv("DEEPSEEK_API_URL"),
)


def do_translate(text, source_lang, target_lang, style, glossary_path=None, iterative=False):
    """翻译流程。"""
    logger.info(
        "开始翻译 | style=%s %s→%s iterative=%s glossary=%s text=%r",
        style, source_lang, target_lang, iterative, glossary_path, text[:50],
    )

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

        logger.info("第 %d/%d 轮：调用 LLM...", i + 1, max_rounds)
        response = llm.invoke(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": current_prompt},
            ]
        )
        translated = response.content
        logger.info("第 %d 轮译文：%r", i + 1, translated[:80])

        scores = score_translation(text, translated, source_lang, target_lang, style)
        avg = sum([scores[k] for k in ["fluency", "accuracy", "style_match"]]) / 3
        logger.info(
            "第 %d 轮评分：流畅 %.1f | 准确 %.1f | 风格 %.1f | 均分 %.1f",
            i + 1, scores["fluency"], scores["accuracy"], scores["style_match"], avg,
        )

        if avg >= 8.0 or not iterative:
            logger.info("结束迭代（共 %d 轮）", i + 1)
            break
        feedback = scores["reason"]
        logger.info("均分 <8，进入下一轮，改进建议：%r", feedback[:80])

    return {"result": translated, "scores": scores, "rounds": i + 1}


def do_rewrite(text, mode):
    """改写流程。"""
    logger.info("开始改写 | mode=%s text=%r", mode, text[:50])
    config = REWRITE_CONFIG[mode]
    response = llm.invoke(
        [
            {"role": "system", "content": config["role"]},
            {"role": "user", "content": f"{config['instruction']}\n\n原文：\n{text}"},
        ]
    )
    logger.info("改写完成：%r", response.content[:80])
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
