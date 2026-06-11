import gradio as gr
from htr_pipeline import HTRPipeline
import tempfile
from PIL import Image

print("Загрузка моделей в память, пожалуйста, подождите...")
pipeline = HTRPipeline(multiscale=True)


def recognize_text(image, use_multiscale):
    if image is None:
        return "Пожалуйста, загрузите изображение.", "", None, "Слов найдено: —"

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp:
        pil_img = Image.fromarray(image)
        pil_img.save(temp.name)
        temp_path = temp.name

    try:
        raw_text, corrected_text, debug_img, word_count = pipeline.process_image(
            temp_path,
            use_multiscale=use_multiscale
        )
    except Exception as e:
        return f"Произошла ошибка при распознавании: {str(e)}", "", None, "Ошибка"

    mode_label = "мультимасштаб" if use_multiscale else "стандарт"
    stats = f"📦 Найдено регионов: {word_count} ({mode_label})"
    return raw_text, corrected_text, debug_img, stats


# --------------------------------------------------------------------------
# Интерфейс Gradio (совместим с Gradio 4.x и 6.x)
# --------------------------------------------------------------------------
with gr.Blocks() as demo:
    gr.Markdown(
        """
        # ✍️ HTR: Распознавание русской рукописи
        Загрузите фотографию или скан страницы с русским рукописным текстом.
        Система найдёт слова, распознает их и исправит опечатки через Яндекс Спеллер.

        > **Модель:** `kazars24/trocr-base-handwritten-ru` (кириллица, HKR датасет)  
        > **Детектор:** EasyOCR/CRAFT с мультимасштабным поиском + NMS
        """
    )

    with gr.Row():
        with gr.Column(scale=1):
            input_image = gr.Image(label="📷 Загрузите фото рукописного текста", height=350)
            multiscale_toggle = gr.Checkbox(
                label="🔍 Мультимасштабная детекция (медленнее, но ловит бледные/слитные слова)",
                value=True
            )
            run_btn = gr.Button("▶ Распознать", variant="primary", size="lg")
            stats_label = gr.Textbox(
                label="Статистика",
                value="Слов найдено: —",
                interactive=False
            )

        with gr.Column(scale=1):
            debug_image = gr.Image(label="🟩 Найденные регионы (зелёные рамки)", height=350)

    with gr.Row():
        raw_output = gr.Textbox(
            label="📄 Сырой распознанный текст (до коррекции)",
            lines=8
        )
        corrected_output = gr.Textbox(
            label="✅ Текст после Яндекс Спеллера",
            lines=8
        )

    run_btn.click(
        fn=recognize_text,
        inputs=[input_image, multiscale_toggle],
        outputs=[raw_output, corrected_output, debug_image, stats_label]
    )

    gr.Markdown(
        """
        ---
        **Советы:**
        - Если слова всё ещё пропадают — убедитесь что чекбокс **Мультимасштабная детекция** включён
        - Для очень бледного текста попробуйте сфотографировать при лучшем освещении
        - Оптимальное разрешение изображения: от 1000px по длинной стороне
        """
    )


if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860, inbrowser=True)
