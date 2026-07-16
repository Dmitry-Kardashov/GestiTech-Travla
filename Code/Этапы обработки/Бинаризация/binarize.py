import cv2
import gradio as gr
import numpy as np


def remove_noise_by_contours(binary_img, min_area):
    if min_area <= 0:
        return binary_img
    contours, _ = cv2.findContours(
        binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    clean_mask = np.zeros_like(binary_img)
    for cnt in contours:
        if cv2.contourArea(cnt) >= min_area:
            cv2.drawContours(clean_mask, [cnt], -1, 255, thickness=cv2.FILLED)
    return clean_mask


def process_image(
    image,
    filter_d,
    sigma_color,
    sigma_space,
    block_size,
    c_val,
    noise_method,
    morph_size,
    min_area,
):
    if image is None:
        return None, None, None

    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    filtered = cv2.bilateralFilter(
        src=gray, d=int(filter_d), sigmaColor=sigma_color, sigmaSpace=sigma_space
    )

    block_size = int(block_size)
    if block_size % 2 == 0:
        block_size += 1

    binary_inv = cv2.adaptiveThreshold(
        src=filtered,
        maxValue=255,
        adaptiveMethod=cv2.ADAPTIVE_THRESH_MEAN_C,
        thresholdType=cv2.THRESH_BINARY_INV,
        blockSize=block_size,
        C=c_val,
    )

    cleaned = binary_inv.copy()
    if noise_method == "Морфологическое открытие (Быстро)":
        if morph_size > 0:
            kernel = cv2.getStructuringElement(
                cv2.MORPH_RECT, (int(morph_size), int(morph_size))
            )
            cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel)
    elif noise_method == "Фильтрация по площади (Чисто)":
        cleaned = remove_noise_by_contours(cleaned, min_area)

    return filtered, binary_inv, cleaned


with gr.Blocks(title="Адаптивная бинаризация плат") as demo:
    gr.Markdown("# Линейный пайплайн обработки изображений")

    # Единый ряд для ВСЕХ элементов интерфейса
    with gr.Row():
        
        # 1. Колонка управления (Исходник + Ползунки)
        with gr.Column(scale=2, min_width=250):
            input_img = gr.Image(type="numpy", label="1. Исходное")

            with gr.Accordion("Bilateral Filter", open=False):
                slider_d = gr.Slider(1, 25, value=9, step=1, label="Диаметр (d)")
                slider_sigma_color = gr.Slider(1, 200, value=75, step=5, label="Sigma Color")
                slider_space = gr.Slider(1, 200, value=75, step=5, label="Sigma Space")

            with gr.Accordion("Адаптивный Порог", open=True):
                slider_block = gr.Slider(3, 99, value=9, step=2, label="Размер блока")
                slider_c = gr.Slider(-20, 20, value=2, step=1, label="Константа C")

            with gr.Accordion("Очистка от шума", open=True):
                radio_method = gr.Radio(
                    choices=["Без очистки", "Морфологическое открытие (Быстро)", "Фильтрация по площади (Чисто)"],
                    value="Морфологическое открытие (Быстро)",
                    label="Метод",
                )
                slider_morph = gr.Slider(1, 7, value=4, step=1, label="Ядро морфологии")
                slider_area = gr.Slider(0, 500, value=250, step=1, label="Мин. площадь (px)")

            btn = gr.Button("Обработать", variant="primary")

        # 2. Колонка: Фильтрация
        with gr.Column(scale=2, min_width=200):
            output_filtered = gr.Image(label="2. Сглаживание (Bilateral)")

        # 3. Колонка: Шумная бинаризация
        with gr.Column(scale=2, min_width=200):
            output_binary = gr.Image(label="3. Порог (Негатив)")

        # 4. Колонка: Итоговый чистый результат
        with gr.Column(scale=2, min_width=200):
            output_cleaned = gr.Image(label="4. Итог (Без шума)")

    # Логика работы
    inputs = [
        input_img, slider_d, slider_sigma_color, slider_space,
        slider_block, slider_c, radio_method, slider_morph, slider_area
    ]
    outputs = [output_filtered, output_binary, output_cleaned]

    btn.click(fn=process_image, inputs=inputs, outputs=outputs)
    for slider in inputs[1:]:
        if hasattr(slider, "change"):
            slider.change(fn=process_image, inputs=inputs, outputs=outputs)

if __name__ == "__main__":
    demo.launch()
