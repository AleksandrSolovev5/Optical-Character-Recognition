import os
os.environ["HF_HUB_VERBOSITY"] = "error"
import logging
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
import warnings
warnings.filterwarnings("ignore", message=".*unauthenticated requests.*")
warnings.filterwarnings("ignore", category=UserWarning)

import gradio as gr
from htr_pipeline import HTRPipeline
import tempfile
from PIL import Image

print("Загрузка...")
pipeline = HTRPipeline()


def recognize_text(image):
    if image is None:
        return "Пожалуйста, загрузите изображение.", "", None

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp:
        pil_img = Image.fromarray(image)
        pil_img.save(temp.name)
        temp_path = temp.name

    try:
        raw_text, corrected_text, debug_img, word_count = pipeline.process_image(temp_path)
    except Exception as e:
        return f"Произошла ошибка при распознавании: {str(e)}", "", None

    return raw_text, corrected_text, debug_img



with gr.Blocks() as demo:
    gr.Markdown(
        f"""
        # Распознавание рукописного текста (HTR)

        Загрузите изображение страницы с рукописным текстом на русском языке.
        Система выполнит детекцию слов, распознавание символов и коррекцию орфографии.

        **Используемая модель распознавания:** `{pipeline.model_name}`
        """
    )

    with gr.Row():
        with gr.Column(scale=1):
            input_image = gr.Image(
                label="Изображение для распознавания",
                height=350,
                sources=["upload"]
            )
            run_btn = gr.Button("Распознать", variant="primary", size="lg")

        with gr.Column(scale=1):
            debug_image = gr.Image(label="Результат сегментации слов", height=350)

    with gr.Row():
        raw_output = gr.Textbox(
            label="Распознанный текст (без автокоррекции)",
            lines=8
        )
        corrected_output = gr.Textbox(
            label="Результат после автокоррекции",
            lines=8
        )

    run_btn.click(
        fn=recognize_text,
        inputs=[input_image],
        outputs=[raw_output, corrected_output, debug_image]
    )


if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860, inbrowser=True)
