import gradio as gr
from translator import do_translate, do_rewrite


def translate_ui(text, source, target, style, iterative):
    if not text.strip():
        return "请输入文本", ""
    result = do_translate(text, source, target, style, iterative=iterative)
    score_text = f"流畅度: {result['scores']['fluency']} | 准确度: {result['scores']['accuracy']} | 风格匹配: {result['scores']['style_match']}"
    return result["result"], score_text


with gr.Blocks(title="多语言翻译工具") as app:
    gr.Markdown("# 多语言翻译与文本改写工具")
    with gr.Row():
        with gr.Column():
            text_in = gr.Textbox(label="输入文本", lines=5)
            source = gr.Dropdown(
                ["中文", "English", "日本語", "한국어", "Français"], label="源语言", value="中文"
            )
            target = gr.Dropdown(
                ["中文", "English", "日本語", "한국어", "Français"],
                label="目标语言",
                value="English",
            )
            style = gr.Dropdown(
                ["literal", "free", "literary", "business"], label="翻译风格", value="free"
            )
            iterative = gr.Checkbox(label="迭代优化")
            btn = gr.Button("翻译", variant="primary")
        with gr.Column():
            text_out = gr.Textbox(label="译文", lines=5)
            score_out = gr.Text(label="质量评分")
    btn.click(
        fn=translate_ui,
        inputs=[text_in, source, target, style, iterative],
        outputs=[text_out, score_out],
    )

app.launch()
