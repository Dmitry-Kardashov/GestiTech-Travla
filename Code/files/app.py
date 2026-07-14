import cv2
import numpy as np
import os
import gradio as gr

# ==========================================
# ⚙️ БЛОК НАСТРОЕК (для веб-интерфейса)
# ==========================================
OUTPUT_DIR_ROOT = 'debugging_inspection'
os.makedirs(OUTPUT_DIR_ROOT, exist_ok=True)

def align_images_orb(img_gerber, img_pcb):
    """
    Автоматическое выравнивание платы по герберу с помощью ключевых точек ORB.
    """
    gray_gerber = cv2.cvtColor(img_gerber, cv2.COLOR_BGR2GRAY)
    gray_pcb = cv2.cvtColor(img_pcb, cv2.COLOR_BGR2GRAY)

    orb = cv2.ORB_create(nfeatures=5000)
    
    kp_gerber, des_gerber = orb.detectAndCompute(gray_gerber, None)
    kp_pcb, des_pcb = orb.detectAndCompute(gray_pcb, None)

    if des_gerber is None or des_pcb is None:
        raise ValueError("Не удалось извлечь дескрипторы точек. Проверь входные изображения.")

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des_gerber, des_pcb)
    
    matches = sorted(matches, key=lambda x: x.distance)
    good_matches = matches[:100]

    if len(good_matches) < 10:
        raise ValueError(f"Слишком мало общих точек соприкосновения ({len(good_matches)}). Выравнивание невозможно.")

    pts_gerber = np.float32([kp_gerber[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    pts_pcb = np.float32([kp_pcb[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    H, mask = cv2.findHomography(pts_pcb, pts_gerber, cv2.RANSAC, 5.0)
    
    gh, gw = img_gerber.shape[:2]
    img_pcb_aligned = cv2.warpPerspective(img_pcb, H, (gw, gh))
    
    return img_pcb_aligned

def binarize_pcb(img_aligned, block_size, c_val):
    """ 
    Выделение меди БЕЗ фильтра Гаусса. 
    Используется обычное локальное среднее (MEAN) для бинаризации.
    """
    # Выделяем медь через разность каналов (без предварительного размытия)
    b, g, r = cv2.split(img_aligned)
    diff = cv2.subtract(r, g)
    diff = cv2.normalize(diff, None, 0, 255, cv2.NORM_MINMAX)
    
    # Корректируем размер блока (обязан быть нечетным и > 1)
    if block_size % 2 == 0:
        block_size += 1
    if block_size < 3:
        block_size = 3

    # Адаптивная бинаризация по СРЕДНЕМУ (ADAPTIVE_THRESH_MEAN_C) вместо ГАУССА
    thresh = cv2.adaptiveThreshold(
        diff, 255, cv2.ADAPTIVE_THRESH_MEAN_C, 
        cv2.THRESH_BINARY, block_size, c_val
    )
    
    # Обычная морфологическая очистка мелких изолированных пикселей
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    return thresh

def inspect_pcb_interface(gerber_img, pcb_img, block_size, c_val, min_area):
    """
    Основная функция-обработчик для веб-интерфейса.
    """
    if gerber_img is None or pcb_img is None:
        return None, None, "Пожалуйста, загрузите оба изображения."

    img_gerber = cv2.cvtColor(gerber_img, cv2.COLOR_RGB2BGR)
    img_pcb = cv2.cvtColor(pcb_img, cv2.COLOR_RGB2BGR)

    status_msg = "Статус: Начинаем обработку...\n"
    
    try:
        # 1. Выравнивание
        img_pcb_aligned = align_images_orb(img_gerber, img_pcb)
        cv2.imwrite(os.path.join(OUTPUT_DIR_ROOT, 'step1_pcb_aligned.jpg'), img_pcb_aligned)
        status_msg += "[1/3] Выравнивание слоев успешно завершено.\n"
        
        # 2. Бинаризация (Метод среднего)
        gerber_gray = cv2.cvtColor(img_gerber, cv2.COLOR_BGR2GRAY)
        _, gerber_bin = cv2.threshold(gerber_gray, 127, 255, cv2.THRESH_BINARY)
        
        pcb_bin = binarize_pcb(img_pcb_aligned, int(block_size), int(c_val))
        cv2.imwrite(os.path.join(OUTPUT_DIR_ROOT, 'step2_pcb_binarized.jpg'), pcb_bin)
        status_msg += "[2/3] Сегментация меди методом локального среднего завершена.\n"

        # Наложение маски зоны интереса
        kernel_roi = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
        roi_mask = cv2.dilate(gerber_bin, kernel_roi)

        gr_active = cv2.bitwise_and(gerber_bin, roi_mask)
        pcb_active = cv2.bitwise_and(pcb_bin, roi_mask)

        # 3. Дифференциальный анализ
        kernel_clean = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        
        missing_copper = cv2.bitwise_and(gr_active, cv2.bitwise_not(pcb_active))
        missing_copper = cv2.morphologyEx(missing_copper, cv2.MORPH_OPEN, kernel_clean)

        excess_copper = cv2.bitwise_and(pcb_active, cv2.bitwise_not(gr_active))
        excess_copper = cv2.morphologyEx(excess_copper, cv2.MORPH_OPEN, kernel_clean)

        output_visual = img_pcb_aligned.copy()

        # Отрисовка дефектов
        contours_missing, _ = cv2.findContours(missing_copper, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        breaks_count = 0
        for c in contours_missing:
            if cv2.contourArea(c) > min_area: 
                breaks_count += 1
                x, y, wc, hc = cv2.boundingRect(c)
                cv2.rectangle(output_visual, (x - 2, y - 2), (x + wc + 2, y + hc + 2), (0, 0, 255), 2)
                cv2.putText(output_visual, f"Break #{breaks_count}", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

        contours_excess, _ = cv2.findContours(excess_copper, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        extras_count = 0
        for c in contours_excess:
            if cv2.contourArea(c) > min_area:
                extras_count += 1
                x, y, wc, hc = cv2.boundingRect(c)
                cv2.rectangle(output_visual, (x - 2, y - 2), (x + wc + 2, y + hc + 2), (255, 0, 0), 2)
                cv2.putText(output_visual, f"Extra #{extras_count}", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)

        result_path = os.path.join(OUTPUT_DIR_ROOT, 'PCB_INSPECTION_RESULT.jpg')
        cv2.imwrite(result_path, output_visual)
        
        status_msg += f"[3/3] Анализ завершен!\nОбрывов: {breaks_count}\nИзлишков: {extras_count}"
        
        return cv2.cvtColor(output_visual, cv2.COLOR_BGR2RGB), cv2.cvtColor(pcb_bin, cv2.COLOR_GRAY2RGB), status_msg

    except Exception as e:
        return None, None, f"[ОШИБКА]: {str(e)}"

# ==========================================
# 🖥️ ИНТЕРФЕЙС GRADIO
# ==========================================
theme = gr.themes.Soft(primary_hue="blue", secondary_hue="gray")

with gr.Blocks(theme=theme, title="PCB Inspection (No Gauss)") as demo:
    gr.Markdown(
        """
        # 🔍 PCB Дефектоскоп (Без фильтрации Гаусса)
        Использует прямое попиксельное вычитание каналов и жесткий адаптивный порог по среднему арифметическому значению окрестности. Размытие краев дорожек полностью отключено.
        """
    )
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 📥 Входные данные")
            gerber_input = gr.Image(label="Шаблон (GERBER)", type="numpy")
            pcb_input = gr.Image(label="Фото платы (PCB)", type="numpy")
            
        with gr.Column(scale=1):
            gr.Markdown("### ⚙️ Параметры адаптивного порога (Mean)")
            block_size_slider = gr.Slider(
                minimum=3, maximum=101, step=2, value=51, 
                label="Размер окна анализа (Block Size)"
            )
            c_val_slider = gr.Slider(
                minimum=-30, maximum=30, step=1, value=-15, 
                label="Чувствительность порога (Constant C)"
            )
            min_area_slider = gr.Slider(
                minimum=10, maximum=1000, step=10, value=120, 
                label="Игнорировать дефекты меньше (Min Area пикс.)"
            )
            submit_btn = gr.Button("🚀 Начать дефектоскопию", variant="primary")
            
        with gr.Column(scale=1.2):
            gr.Markdown("### 📤 Вывод")
            bin_output = gr.Image(label="Бинарная маска меди (Окно среднего)")
            result_output = gr.Image(label="Карта дефектов")
            status_output = gr.Textbox(label="Статус", interactive=False, lines=4)

    submit_btn.click(
        fn=inspect_pcb_interface,
        inputs=[gerber_input, pcb_input, block_size_slider, c_val_slider, min_area_slider],
        outputs=[result_output, bin_output, status_output]
    )

if __name__ == "__main__":
    demo.launch(share=False)