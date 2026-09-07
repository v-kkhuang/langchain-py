import json


class Glossary:
    """术语表：维护术语的统一翻译映射。"""

    def __init__(self):
        self.terms = {}  # {"源语言术语": "目标语言翻译"}

    def add(self, source: str, target: str):
        self.terms[source] = target

    def load_from_file(self, filepath: str):
        with open(filepath, "r", encoding="utf-8") as f:
            self.terms = json.load(f)

    def format_for_prompt(self) -> str:
        if not self.terms:
            return ""
        lines = ["请确保以下术语翻译一致："]
        for src, tgt in self.terms.items():
            lines.append(f"  {src} → {tgt}")
        return "\n".join(lines)
