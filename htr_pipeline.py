import cv2
import easyocr
import torch
import numpy as np
import requests
from PIL import Image, ImageFilter, ImageEnhance
from transformers import TrOCRProcessor, VisionEncoderDecoderModel


class HTRPipeline:
    def __init__(self, multiscale: bool = True):
        print("Инициализация HTR пайплайна...")
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Используемое устройство: {self.device}")

        # Флаг мультимасштабной детекции
        self.multiscale = multiscale
        # Масштабы для мультимасштабного прохода (3.0 ловит самые бледные слова)
        self.mag_ratios = [1.5, 2.0, 2.5, 3.0]
        # IoU-порог для NMS (0.5 = агрессивное слияние дубликатов с разных масштабов)
        self.nms_iou_threshold = 0.5
        # Порог вложенности: если маленький бокс на 60%+ внутри большого — сливаем
        self.containment_threshold = 0.6

        print("Загрузка детектора текста (EasyOCR/CRAFT)...")
        self.detector = easyocr.Reader(['en'], gpu=torch.cuda.is_available())

        # ---------------------------------------------------------------
        # Модель распознавания: kazars24/trocr-base-handwritten-ru
        # Обучена на 57,000+ сегментах кириллицы (HKR датасет).
        # Выдаёт корректный кириллический текст для русской прописи.
        # Примечание: akushsky/trocr-large-handwritten-ru была отклонена —
        # она выдаёт транслитерацию латиницей вместо кириллицы.
        # ---------------------------------------------------------------
        model_name = "kazars24/trocr-base-handwritten-ru"

        print(f"Загрузка модели распознавания: {model_name}...")
        self.processor = TrOCRProcessor.from_pretrained(model_name)
        self.model = VisionEncoderDecoderModel.from_pretrained(model_name).to(self.device)

        self.model.eval()
        print(f"✅ Модель {model_name} загружена!")
        print(f"HTR пайплайн успешно загружен! Мультимасштаб: {'ВКЛ' if self.multiscale else 'ВЫКЛ'}")

    # ------------------------------------------------------------------
    # УЛУЧШЕНИЕ 2: Мультимасштабный детектор с NMS
    # ------------------------------------------------------------------

    def _boxes_to_xyxy(self, boxes):
        """Конвертирует список poly-боксов [[x1,y1],[x2,y1],...] в [x_min,y_min,x_max,y_max]."""
        result = []
        for b in boxes:
            b = np.array(b)
            x_min, x_max = int(np.min(b[:, 0])), int(np.max(b[:, 0]))
            y_min, y_max = int(np.min(b[:, 1])), int(np.max(b[:, 1]))
            result.append([x_min, y_min, x_max, y_max])
        return result

    def _should_merge(self, boxA, boxB):
        """
        Определяет, нужно ли слить два бокса.
        Использует два критерия:
        1. IoU > nms_iou_threshold (классический NMS)
        2. Containment: если маленький бокс на 60%+ внутри большого
           (ловит случаи, когда один масштаб нашёл часть слова, а другой — целиком)
        """
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])
        inter = max(0, xB - xA) * max(0, yB - yA)
        if inter == 0:
            return False
        areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
        areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
        union = float(areaA + areaB - inter)
        iou = inter / union if union > 0 else 0
        # Containment: какая доля меньшего бокса лежит внутри пересечения?
        min_area = min(areaA, areaB)
        containment = inter / float(min_area) if min_area > 0 else 0
        return iou > self.nms_iou_threshold or containment > self.containment_threshold

    def _nms_boxes(self, boxes_xyxy):
        """
        Non-Maximum Suppression: убирает дублирующиеся боксы.
        Боксы, перекрывающиеся более чем на nms_iou_threshold, сливаются в один
        (берётся объединяющий bounding box, а не просто удаляется один из двух).
        """
        if not boxes_xyxy:
            return []

        merged = list(boxes_xyxy)
        changed = True
        while changed:
            changed = False
            result = []
            used = [False] * len(merged)
            for i in range(len(merged)):
                if used[i]:
                    continue
                current = list(merged[i])
                for j in range(i + 1, len(merged)):
                    if used[j]:
                        continue
                    if self._should_merge(current, merged[j]):
                        # Сливаем боксы в объединяющий прямоугольник
                        current[0] = min(current[0], merged[j][0])
                        current[1] = min(current[1], merged[j][1])
                        current[2] = max(current[2], merged[j][2])
                        current[3] = max(current[3], merged[j][3])
                        used[j] = True
                        changed = True
                result.append(current)
                used[i] = True
            merged = result

        return merged

    def _filter_small_boxes(self, boxes_xyxy, img_shape):
        """
        Отсеивает слишком маленькие боксы (одиночные буквы, мусор).
        Минимальная ширина — 1.5% от ширины изображения.
        """
        img_h, img_w = img_shape[:2]
        min_width = int(img_w * 0.015)
        min_height = int(img_h * 0.01)
        filtered = []
        for box in boxes_xyxy:
            w = box[2] - box[0]
            h = box[3] - box[1]
            if w >= min_width and h >= min_height:
                filtered.append(box)
        removed = len(boxes_xyxy) - len(filtered)
        if removed > 0:
            print(f"    Отфильтровано мелких фрагментов: {removed}")
        return filtered

    def _detect_multiscale(self, img):
        """
        Запускает детектор CRAFT с разными mag_ratio и объединяет результаты через NMS.
        Все масштабы используют стандартные пороги — пониженные пороги
        создают фрагменты одиночных букв.
        """
        all_boxes_xyxy = []

        for mag_ratio in self.mag_ratios:
            try:
                result = self.detector.detect(img, mag_ratio=mag_ratio)
                horizontal_list, _ = result
                boxes = horizontal_list[0] if horizontal_list else []
                print(f"    mag_ratio={mag_ratio}: найдено {len(boxes)} боксов")
                for box in boxes:
                    x_min, x_max, y_min, y_max = map(int, box)
                    all_boxes_xyxy.append([x_min, y_min, x_max, y_max])
            except Exception as e:
                print(f"  ⚠️ Ошибка при mag_ratio={mag_ratio}: {e}")

        print(f"    Всего до NMS: {len(all_boxes_xyxy)} боксов")
        # Убираем мелкие фрагменты
        all_boxes_xyxy = self._filter_small_boxes(all_boxes_xyxy, img.shape)
        # Убираем дублирующиеся боксы
        unique_boxes = self._nms_boxes(all_boxes_xyxy)
        print(f"    После NMS: {len(unique_boxes)} уникальных боксов")

        # Конвертируем обратно в poly-формат
        poly_boxes = []
        for (x_min, y_min, x_max, y_max) in unique_boxes:
            poly_boxes.append([
                [x_min, y_min], [x_max, y_min],
                [x_max, y_max], [x_min, y_max]
            ])
        return poly_boxes

    def _detect_single(self, img):
        """Стандартная однопроходная детекция (быстрая)."""
        result = self.detector.detect(img, mag_ratio=1.5)
        horizontal_list, _ = result
        boxes = horizontal_list[0] if horizontal_list else []
        poly_boxes = []
        for box in boxes:
            x_min, x_max, y_min, y_max = map(int, box)
            poly_boxes.append([
                [x_min, y_min], [x_max, y_min],
                [x_max, y_max], [x_min, y_max]
            ])
        return poly_boxes

    # ------------------------------------------------------------------
    # УЛУЧШЕНИЕ 3: Умная предобработка кропов перед TrOCR
    # ------------------------------------------------------------------

    def _preprocess_crop(self, pil_img: Image.Image) -> Image.Image:
        """
        Предобработка вырезанного слова перед подачей в TrOCR.
        
        ВАЖНО: UnsharpMask и усиление контраста были убраны —
        они искажали входные данные и ухудшали распознавание.
        Модель обучена на естественных сканах, поэтому лучше
        подавать изображение без модификаций.
        """
        return pil_img

    # ------------------------------------------------------------------
    # Сортировка боксов по строкам (без изменений)
    # ------------------------------------------------------------------

    def sort_boxes(self, boxes):
        """
        Группировка боксов по строкам.
        Сортируем по верхнему краю (y_min), чтобы избежать прыжков слов с высокими буквами.
        """
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

    # ------------------------------------------------------------------
    # Основной метод обработки изображения
    # ------------------------------------------------------------------

    def process_image(self, image_path, use_multiscale: bool = None):
        """
        Обрабатывает изображение и возвращает (сырой текст, исправленный текст, debug-изображение, кол-во слов).
        
        Args:
            image_path: путь к изображению
            use_multiscale: переопределить флаг мультимасштаба (None = использовать self.multiscale)
        """
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Не удалось загрузить изображение: {image_path}")

        # Выбираем режим детекции
        if use_multiscale is None:
            use_multiscale = self.multiscale

        if use_multiscale:
            print(f"  🔍 Мультимасштабная детекция (mag_ratio={self.mag_ratios})...")
            poly_boxes = self._detect_multiscale(img)
        else:
            print("  🔍 Стандартная детекция (mag_ratio=1.5)...")
            poly_boxes = self._detect_single(img)

        total_words = len(poly_boxes)
        print(f"  📦 Найдено регионов: {total_words}")

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

                # Отступы: 25% по Y (для ascenders/descenders), 10% по X (для крайних букв)
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

                # УЛУЧШЕНИЕ 3: предобработка кропа перед TrOCR
                pil_img = self._preprocess_crop(pil_img)

                line_images.append(pil_img)

            if not line_images:
                continue

            # Батчинг: подаём слова построчно
            with torch.no_grad():
                pixel_values = self.processor(
                    images=line_images, return_tensors="pt"
                ).pixel_values.to(self.device)

                generated_ids = self.model.generate(
                    pixel_values,
                    max_new_tokens=32,
                    max_length=None,      # Убирает конфликт с max_length из конфига модели
                    num_beams=4,          # Beam search для лучшего качества
                    early_stopping=True
                )

            words = self.processor.batch_decode(generated_ids, skip_special_tokens=True)
            recognized_line = " ".join(words)
            recognized_text_lines.append(recognized_line)

        raw_text = "\n".join(recognized_text_lines)
        corrected_text = self.correct_text_yandex(raw_text)

        img_debug_rgb = cv2.cvtColor(img_debug, cv2.COLOR_BGR2RGB)
        return raw_text, corrected_text, img_debug_rgb, total_words

    # ------------------------------------------------------------------
    # Яндекс Спеллер (без изменений)
    # ------------------------------------------------------------------

    def correct_text_yandex(self, text):
        if not text.strip():
            return text
        try:
            print("  📝 Отправка в Яндекс Спеллер...")
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
            print(f"  ⚠️  Ошибка Яндекс Спеллера: {e}")
            return text
