import cv2
import torch
import numpy as np
import requests
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel


class HTRPipeline:
    def __init__(self):
        print("Инициализация HTR...")
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Устройство: {self.device}")

        from doctr.models import detection_predictor
        self.doctr_detector = detection_predictor(
            arch='db_resnet50',
            pretrained=True,
            assume_straight_pages=True
        )
        self.doctr_detector.model.to(self.device)

        model_name = "kazars24/trocr-base-handwritten-ru"
        self.processor = TrOCRProcessor.from_pretrained(model_name)
        self.model = VisionEncoderDecoderModel.from_pretrained(model_name).to(self.device)
        self.model.eval()
        print("Пайплайн HTR готов.")

    def _detect_doctr(self, img):
        h, w = img.shape[:2]
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        with torch.no_grad():
            result = self.doctr_detector([img_rgb])

        poly_boxes = []
        if len(result) > 0:
            page_boxes = result[0]['words']

            for box in page_boxes:
                x_min = int(box[0] * w)
                y_min = int(box[1] * h)
                x_max = int(box[2] * w)
                y_max = int(box[3] * h)

                box_w = x_max - x_min
                box_h = y_max - y_min
                if (box_w < w * 0.02 and box_h < h * 0.02) or box_h < h * 0.008 or box_w < w * 0.008:
                    continue

                poly_boxes.append([
                    [x_min, y_min], [x_max, y_min],
                    [x_max, y_max], [x_min, y_max]
                ])

        return poly_boxes

    def sort_boxes(self, boxes):
        if not boxes:
            return []

        processed = []
        for b in boxes:
            b = np.array(b)
            x_min, x_max = np.min(b[:, 0]), np.max(b[:, 0])
            y_min, y_max = np.min(b[:, 1]), np.max(b[:, 1])
            cy = (y_min + y_max) / 2
            processed.append({
                'box': b,
                'x_min': x_min, 'x_max': x_max,
                'y_min': y_min, 'y_max': y_max,
                'cy': cy
            })

        processed.sort(key=lambda item: item['y_min'])

        lines = []
        current_line = [processed[0]]

        for i in range(1, len(processed)):
            curr = processed[i]
            line_y_min = np.min([item['y_min'] for item in current_line])
            line_y_max = np.max([item['y_max'] for item in current_line])

            if line_y_min <= curr['cy'] <= line_y_max:
                current_line.append(curr)
            else:
                lines.append(current_line)
                current_line = [curr]

        if current_line:
            lines.append(current_line)

        final_lines = []
        for line in lines:
            line.sort(key=lambda item: item['x_min'])
            final_lines.append([item['box'] for item in line])

        return final_lines

    def process_image(self, image_path):
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Не удалось загрузить изображение: {image_path}")

        print("Детекция...")
        poly_boxes = self._detect_doctr(img)

        total_words = len(poly_boxes)
        print(f"Найдено регионов: {total_words}")

        if total_words == 0:
            img_debug_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            return "", "", img_debug_rgb, 0

        lines_of_boxes = self.sort_boxes(poly_boxes)

        recognized_text_lines = []
        img_debug = img.copy()

        for line_boxes in lines_of_boxes:
            line_images = []

            for box in line_boxes:
                box = np.array(box)
                x_min = max(0, int(np.min(box[:, 0])))
                y_min = max(0, int(np.min(box[:, 1])))
                x_max = int(np.max(box[:, 0]))
                y_max = int(np.max(box[:, 1]))

                cv2.rectangle(img_debug, (x_min, y_min), (x_max, y_max), (0, 220, 90), 2)

                pad_y = int((y_max - y_min) * 0.25)
                pad_x = int((x_max - x_min) * 0.05)

                y_min_pad = max(0, y_min - pad_y)
                y_max_pad = min(img.shape[0], y_max + pad_y)
                x_min_pad = max(0, x_min - pad_x)
                x_max_pad = min(img.shape[1], x_max + pad_x)

                cropped = img[y_min_pad:y_max_pad, x_min_pad:x_max_pad]
                if cropped.size == 0:
                    continue

                cropped_rgb = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(cropped_rgb)
                line_images.append(pil_img)

            if not line_images:
                continue

            with torch.no_grad():
                pixel_values = self.processor(
                    images=line_images, return_tensors="pt"
                ).pixel_values.to(self.device)

                generated_ids = self.model.generate(
                    pixel_values,
                    max_new_tokens=32,
                    max_length=None,
                    num_beams=4,
                    early_stopping=True
                )

            words = self.processor.batch_decode(generated_ids, skip_special_tokens=True)
            recognized_line = " ".join(words)
            recognized_text_lines.append(recognized_line)

        raw_text = "\n".join(recognized_text_lines)
        corrected_text = self.correct_text_yandex(raw_text)

        img_debug_rgb = cv2.cvtColor(img_debug, cv2.COLOR_BGR2RGB)
        return raw_text, corrected_text, img_debug_rgb, total_words

    def correct_text_yandex(self, text):
        if not text.strip():
            return text
        try:
            print("Коррекция текста...")
            url = "https://speller.yandex.net/services/spellservice.json/checkText"
            data = {'text': text, 'lang': 'ru', 'options': 518}
            response = requests.post(url, data=data, timeout=5)

            if response.status_code == 200:
                changes = response.json()
                for change in reversed(changes):
                    if change['s']:
                        suggestion = change['s'][0]
                        pos = change['pos']
                        length = change['len']
                        text = text[:pos] + suggestion + text[pos + length:]
            return text
        except Exception as e:
            print(f"Ошибка проверки орфографии: {e}")
            return text
