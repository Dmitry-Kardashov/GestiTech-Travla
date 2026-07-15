# import cv2
# import numpy as np
# import os
# import gradio as gr

# # ==========================================
# # ⚙️ БЛОК НАСТРОЕК (для веб-интерфейса)
# # ==========================================
# OUTPUT_DIR_ROOT = 'debugging_inspection'
# os.makedirs(OUTPUT_DIR_ROOT, exist_ok=True)

# def align_images_orb(img_gerber, img_pcb):
#     """
#     Автоматическое выравнивание платы по герберу с помощью ключевых точек ORB.
#     """
#     gray_gerber = cv2.cvtColor(img_gerber, cv2.COLOR_BGR2GRAY)
#     gray_pcb = cv2.cvtColor(img_pcb, cv2.COLOR_BGR2GRAY)

#     orb = cv2.ORB_create(nfeatures=5000)
    
#     kp_gerber, des_gerber = orb.detectAndCompute(gray_gerber, None)
#     kp_pcb, des_pcb = orb.detectAndCompute(gray_pcb, None)

#     if des_gerber is None or des_pcb is None:
#         raise ValueError("Не удалось извлечь дескрипторы точек. Проверь входные изображения.")

#     bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
#     matches = bf.match(des_gerber, des_pcb)
    
#     matches = sorted(matches, key=lambda x: x.distance)
#     good_matches = matches[:100]

#     if len(good_matches) < 10:
#         raise ValueError(f"Слишком мало общих точек соприкосновения ({len(good_matches)}). Выравнивание невозможно.")

#     pts_gerber = np.float32([kp_gerber[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
#     pts_pcb = np.float32([kp_pcb[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

#     H, mask = cv2.findHomography(pts_pcb, pts_gerber, cv2.RANSAC, 5.0)
    
#     gh, gw = img_gerber.shape[:2]
#     img_pcb_aligned = cv2.warpPerspective(img_pcb, H, (gw, gh))
    
#     return img_pcb_aligned

# def binarize_pcb(img_aligned, block_size, c_val):
#     """ Адаптивное выделение меди на основе локального контраста с ручными параметрами """
#     b, g, r = cv2.split(img_aligned)
    
#     # Медь сильнее отражает красный канал, подложка — зеленый
#     diff = cv2.subtract(r, g)
#     diff = cv2.normalize(diff, None, 0, 255, cv2.NORM_MINMAX)
    
#     # Убедимся, что размер блока нечетный и больше 1
#     if block_size % 2 == 0:
#         block_size += 1
#     if block_size < 3:
#         block_size = 3

#     # Бинаризация Гаусса с внешними параметрами
#     thresh = cv2.adaptiveThreshold(
#         diff, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
#         cv2.THRESH_BINARY, block_size, c_val
#     )
    
#     # Убираем шумы
#     kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
#     thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
#     thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
#     return thresh

# def inspect_pcb_interface(gerber_img, pcb_img, block_size, c_val, min_area):
#     """
#     Основная функция-обработчик для Gradio интерфейса.
#     Принимает изображения и настраиваемые параметры бинаризации.
#     """
#     if gerber_img is None or pcb_img is None:
#         return None, None, "Пожалуйста, загрузите оба изображения (Шаблон и Фото платы)."

#     # Конвертируем из RGB (Gradio) в BGR (OpenCV)
#     img_gerber = cv2.cvtColor(gerber_img, cv2.COLOR_RGB2BGR)
#     img_pcb = cv2.cvtColor(pcb_img, cv2.COLOR_RGB2BGR)

#     status_msg = "Статус: Начинаем обработку...\n"
    
#     try:
#         # 1. Выравнивание
#         img_pcb_aligned = align_images_orb(img_gerber, img_pcb)
#         cv2.imwrite(os.path.join(OUTPUT_DIR_ROOT, 'step1_pcb_aligned.jpg'), img_pcb_aligned)
#         status_msg += "[1/3] Выравнивание слоев успешно завершено.\n"
        
#         # 2. Бинаризация (с использованием переданных ползунков)
#         gerber_gray = cv2.cvtColor(img_gerber, cv2.COLOR_BGR2GRAY)
#         _, gerber_bin = cv2.threshold(gerber_gray, 127, 255, cv2.THRESH_BINARY)
        
#         pcb_bin = binarize_pcb(img_pcb_aligned, int(block_size), int(c_val))
#         cv2.imwrite(os.path.join(OUTPUT_DIR_ROOT, 'step2_pcb_binarized.jpg'), pcb_bin)
#         status_msg += "[2/3] Сегментация меди завершена на основе выбранных параметров.\n"

#         # Наложение маски
#         kernel_roi = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
#         roi_mask = cv2.dilate(gerber_bin, kernel_roi)

#         gerber_active = cv2.bitwise_and(gerber_bin, roi_mask)
#         pcb_active = cv2.bitwise_and(pcb_bin, roi_mask)

#         # 3. Поиск дефектов
#         kernel_clean = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        
#         # Обрывы
#         missing_copper = cv2.bitwise_and(gerber_active, cv2.bitwise_not(pcb_active))
#         missing_copper = cv2.morphologyEx(missing_copper, cv2.MORPH_OPEN, kernel_clean)

#         # Лишняя медь
#         excess_copper = cv2.bitwise_and(pcb_active, cv2.bitwise_not(gerber_active))
#         excess_copper = cv2.morphologyEx(excess_copper, cv2.MORPH_OPEN, kernel_clean)

#         output_visual = img_pcb_aligned.copy()

#         # Отрисовка обрывов (Красный)
#         contours_missing, _ = cv2.findContours(missing_copper, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#         breaks_count = 0
#         for c in contours_missing:
#             if cv2.contourArea(c) > min_area: 
#                 breaks_count += 1
#                 x, y, wc, hc = cv2.boundingRect(c)
#                 cv2.rectangle(output_visual, (x - 2, y - 2), (x + wc + 2, y + hc + 2), (0, 0, 255), 2)
#                 cv2.putText(output_visual, f"Break #{breaks_count}", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

#         # Отрисовка лишней меди (Синий)
#         contours_excess, _ = cv2.findContours(excess_copper, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#         extras_count = 0
#         for c in contours_excess:
#             if cv2.contourArea(c) > min_area+10:
#                 extras_count += 1
#                 x, y, wc, hc = cv2.boundingRect(c)
#                 cv2.rectangle(output_visual, (x - 2, y - 2), (x + wc + 2, y + hc + 2), (255, 0, 0), 2)
#                 cv2.putText(output_visual, f"Extra #{extras_count}", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)

#         # Сохраняем итоговый результат
#         result_path = os.path.join(OUTPUT_DIR_ROOT, 'PCB_INSPECTION_RESULT.jpg')
#         cv2.imwrite(result_path, output_visual)
        
#         status_msg += f"[3/3] Анализ завершен!\nНайдено обрывов (Break): {breaks_count}\nНайдено излишков (Extra): {extras_count}\n\nРезультат сохранен в '{result_path}'"
        
#         # Конвертируем изображения обратно в RGB для корректного отображения в Gradio
#         output_visual_rgb = cv2.cvtColor(output_visual, cv2.COLOR_BGR2RGB)
#         pcb_bin_rgb = cv2.cvtColor(pcb_bin, cv2.COLOR_GRAY2RGB)
        
#         return output_visual_rgb, pcb_bin_rgb, status_msg

#     except Exception as e:
#         error_msg = f"[ОШИБКА ОБРАБОТКИ]: {str(e)}"
#         print(error_msg)
#         return None, None, error_msg

# # ==========================================
# # 🖥️ ОПИСАНИЕ ИНТЕРФЕЙСА GRADIO
# # ==========================================
# theme = gr.themes.Soft(primary_hue="blue", secondary_hue="gray")

# with gr.Blocks(theme=theme, title="PCB Inspection AI") as demo:
#     gr.Markdown(
#         """
#         # 🔍 Система автоматического контроля дефектов печатных плат
#         Загрузите эталонное изображение (Gerber) и реальное фото вашей платы. Настройте ползунки бинаризации справа, чтобы получить идеальное выделение медных дорожек, убрав лишние шумы и текстолит.
#         """
#     )
    
#     with gr.Row():
#         # Левая колонка: Загрузка файлов
#         with gr.Column(scale=1, min_width=300):
#             gr.Markdown("### 📥 Шаг 1: Входные данные")
#             gerber_input = gr.Image(label="Эталон (GERBER / Шаблон)", type="numpy")
#             pcb_input = gr.Image(label="Фото платы (PCB)", type="numpy")
            
#         # Средняя колонка: Интерактивные настройки
#         with gr.Column(scale=1, min_width=300):
#             gr.Markdown("### ⚙️ Шаг 2: Тонкая настройка бинаризации")
            
#             # Слайдер для размера блока локальной адаптации (должен быть нечетным)
#             block_size_slider = gr.Slider(
#                 minimum=3, maximum=101, step=2, value=51, 
#                 label="Размер локального блока (Adaptive Block Size)", 
#                 info="Большие значения сглаживают перепады освещения, меньшие выделяют мелкие детали."
#             )
            
#             # Слайдер для константы C
#             c_val_slider = gr.Slider(
#                 minimum=-30, maximum=30, step=1, value=-15, 
#                 label="Смещение порога (Constant C)", 
#                 info="Уменьшайте (в минус), если на плате проступает текстура подложки. Увеличивайте, если пропадают дорожки."
#             )
            
#             # Минимальный размер дефекта
#             min_area_slider = gr.Slider(
#                 minimum=10, maximum=1000, step=10, value=120, 
#                 label="Фильтр шума дефектов (Min Area)", 
#                 info="Игнорировать дефекты, площадь которых меньше указанного количества пикселей."
#             )
            
#             submit_btn = gr.Button("🚀 Запустить анализ", variant="primary")
            
#         # Правая колонка: Вывод результатов
#         with gr.Column(scale=1, min_width=300):
#             gr.Markdown("### 📤 Шаг 3: Результаты")
#             result_output = gr.Image(label="Найденные дефекты (Красный: Break / Синий: Extra)")
#             bin_output = gr.Image(label="Ч/Б Маска меди (Результат бинаризации)")
#             status_output = gr.Textbox(label="Лог работы системы", interactive=False, lines=5)

#     # Связываем элементы интерфейса с обработчиком
#     submit_btn.click(
#         fn=inspect_pcb_interface,
#         inputs=[
#             gerber_input, 
#             pcb_input, 
#             block_size_slider, 
#             c_val_slider, 
#             min_area_slider
#         ],
#         outputs=[result_output, bin_output, status_output]
#     )

# if __name__ == "__main__":
#     demo.launch(share=False)




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
    Улучшенное автоматическое выравнивание платы по герберу с предварительным 
    выравниванием гистограммы (CLAHE) для проявления деталей на окисленной меди.
    """
    gray_gerber = cv2.cvtColor(img_gerber, cv2.COLOR_BGR2GRAY)
    gray_pcb = cv2.cvtColor(img_pcb, cv2.COLOR_BGR2GRAY)

    # Применяем локальное выравнивание гистограммы (CLAHE) для улучшения контраста фич
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    gray_gerber_enhanced = clahe.apply(gray_gerber)
    gray_pcb_enhanced = clahe.apply(gray_pcb)

    # Ищем до 7000 точек для более надежного сцепления на сложных платах
    orb = cv2.ORB_create(nfeatures=7000)
    
    kp_gerber, des_gerber = orb.detectAndCompute(gray_gerber_enhanced, None)
    kp_pcb, des_pcb = orb.detectAndCompute(gray_pcb_enhanced, None)

    if des_gerber is None or des_pcb is None:
        raise ValueError("Не удалось извлечь дескрипторы точек. Проверь входные изображения.")

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des_gerber, des_pcb)
    
    matches = sorted(matches, key=lambda x: x.distance)
    good_matches = matches[:150]  # Берем чуть больше точек для стабильности гомографии

    if len(good_matches) < 10:
        raise ValueError(f"Слишком мало общих точек соприкосновения ({len(good_matches)}). Выравнивание невозможно.")

    pts_gerber = np.float32([kp_gerber[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    pts_pcb = np.float32([kp_pcb[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    # RANSAC отфильтрует ошибочные точки на окислах
    H, mask = cv2.findHomography(pts_pcb, pts_gerber, cv2.RANSAC, 5.0)
    
    gh, gw = img_gerber.shape[:2]
    img_pcb_aligned = cv2.warpPerspective(img_pcb, H, (gw, gh))
    
    return img_pcb_aligned

def binarize_pcb(img_aligned, block_size, c_val, lab_weight, erode_iters):
    """ 
    Улучшенное выделение меди, устойчивое к окислам.
    Комбинирует стандартный контраст каналов с цветовым пространством LAB.
    """
    # 1. Классический контраст: Медь сильнее отражает красный канал, подложка — зеленый
    b, g, r = cv2.split(img_aligned)
    diff_rgb = cv2.subtract(r, g)
    diff_rgb = cv2.normalize(diff_rgb, None, 0, 255, cv2.NORM_MINMAX)
    
    # 2. Анализ в пространстве LAB (канал B отлично реагирует на переходы меди/окисла в желтизну/синеву)
    lab = cv2.cvtColor(img_aligned, cv2.COLOR_BGR2LAB)
    l_ch, a_ch, b_ch = cv2.split(lab)
    diff_lab = cv2.normalize(b_ch, None, 0, 255, cv2.NORM_MINMAX)

    # Смешиваем два представления на основе ползунка из интерфейса
    weight = float(lab_weight)
    combined_diff = cv2.addWeighted(diff_rgb, 1.0 - weight, diff_lab, weight, 0)
    combined_diff = cv2.normalize(combined_diff, None, 0, 255, cv2.NORM_MINMAX)
    
    # Корректируем размер блока адаптивного порога
    if block_size % 2 == 0:
        block_size += 1
    if block_size < 3:
        block_size = 3

    # Адаптивная бинаризация по среднему значению
    thresh = cv2.adaptiveThreshold(
        combined_diff, 255, cv2.ADAPTIVE_THRESH_MEAN_C, 
        cv2.THRESH_BINARY, block_size, c_val
    )
    
    # Базовая очистка мелкого мусора
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    # Применение эрозии
    if erode_iters > 0:
        erode_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        thresh = cv2.erode(thresh, erode_kernel, iterations=int(erode_iters))

    return thresh

def inspect_pcb_interface(gerber_img, pcb_img, block_size, c_val, lab_weight, erode_iters, min_area):
    """
    Основная функция-обработчик для веб-интерфейса.
    """
    if gerber_img is None or pcb_img is None:
        return None, None, "Пожалуйста, загрузите оба изображения."

    img_gerber = cv2.cvtColor(gerber_img, cv2.COLOR_RGB2BGR)
    img_pcb = cv2.cvtColor(pcb_img, cv2.COLOR_RGB2BGR)

    status_msg = "Статус: Начинаем обработку...\n"
    
    try:
        # 1. Повышение точности выравнивания
        img_pcb_aligned = align_images_orb(img_gerber, img_pcb)
        cv2.imwrite(os.path.join(OUTPUT_DIR_ROOT, 'step1_pcb_aligned.jpg'), img_pcb_aligned)
        status_msg += "[1/3] Улучшенное выравнивание слоев (с адаптивным контрастом CLAHE) завершено.\n"
        
        # 2. Продвинутая бинаризация
        gerber_gray = cv2.cvtColor(img_gerber, cv2.COLOR_BGR2GRAY)
        _, gerber_bin = cv2.threshold(gerber_gray, 127, 255, cv2.THRESH_BINARY)
        
        pcb_bin = binarize_pcb(img_pcb_aligned, int(block_size), int(c_val), lab_weight, int(erode_iters))
        cv2.imwrite(os.path.join(OUTPUT_DIR_ROOT, 'step2_pcb_binarized.jpg'), pcb_bin)
        status_msg += "[2/3] Сегментация меди (RGB + LAB анализ окисления) завершена.\n"

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
        
        status_msg += f"[3/3] Анализ завершен!\nОбрывов (Break): {breaks_count}\nИзлишков (Extra): {extras_count}"
        
        return cv2.cvtColor(output_visual, cv2.COLOR_BGR2RGB), cv2.cvtColor(pcb_bin, cv2.COLOR_GRAY2RGB), status_msg

    except Exception as e:
        return None, None, f"[ОШИБКА]: {str(e)}"

# ==========================================
# 🖥️ ИНТЕРФЕЙС GRADIO
# ==========================================
theme = gr.themes.Soft(primary_hue="blue", secondary_hue="gray")

with gr.Blocks(theme=theme, title="PCB Advanced Inspection") as demo:
    gr.Markdown(
        """
        # 🔍 PCB Дефектоскоп (Оптимизация под окисленную медь)
        В алгоритм добавлено предварительное проявление скрытых фич выравнивания через **CLAHE** и гибридный анализ цветовых пространств **RGB + LAB** для обнаружения потемневшей меди.
        """
    )
    
    with gr.Row():
        # Входные изображения
        with gr.Column(scale=1):
            gr.Markdown("### 📥 Входные данные")
            gerber_input = gr.Image(label="Шаблон (GERBER)", type="numpy")
            pcb_input = gr.Image(label="Фото платы (PCB)", type="numpy")
            
        # Панель управления (Ползунки)
        with gr.Column(scale=1):
            gr.Markdown("### ⚙️ Параметры фильтрации")
            
            with gr.Group():
                gr.Markdown("**1. Адаптивная бинаризация**")
                block_size_slider = gr.Slider(
                    minimum=3, maximum=101, step=2, value=51, 
                    label="Размер окна (Block Size)"
                )
                c_val_slider = gr.Slider(
                    minimum=-30, maximum=30, step=1, value=-15, 
                    label="Чувствительность (Constant C)"
                )
            
            with gr.Group():
                gr.Markdown("**2. Борьба с окислами (LAB Цветокоррекция)**")
                lab_weight_slider = gr.Slider(
                    minimum=0.0, maximum=1.0, step=0.05, value=0.4, 
                    label="Вес LAB-канала (Окисленная медь)",
                    info="0.0 — только стандартный контраст. Смещайте к 1.0, чтобы проявить темные, окисленные или зеленые участки меди."
                )
            
            with gr.Group():
                gr.Markdown("**3. Морфология и Фильтры шума**")
                erode_slider = gr.Slider(
                    minimum=0, maximum=10, step=1, value=0, 
                    label="Итерации эрозии (Erosion)"
                )
                min_area_slider = gr.Slider(
                    minimum=10, maximum=1000, step=10, value=120, 
                    label="Минимальная площадь дефекта (Min Area)"
                )
                
            submit_btn = gr.Button("🚀 Запустить анализ", variant="primary")
            
        # Вывод результатов
        with gr.Column(scale=1.2):
            gr.Markdown("### 📤 Результаты")
            result_output = gr.Image(label="Карта дефектов")
            bin_output = gr.Image(label="Ч/Б Маска меди (Окислы учтены)")
            status_output = gr.Textbox(label="Статус", interactive=False, lines=4)

    submit_btn.click(
        fn=inspect_pcb_interface,
        inputs=[
            gerber_input, 
            pcb_input, 
            block_size_slider, 
            c_val_slider, 
            lab_weight_slider,
            erode_slider, 
            min_area_slider
        ],
        outputs=[result_output, bin_output, status_output]
    )

if __name__ == "__main__":
    demo.launch(share=False)