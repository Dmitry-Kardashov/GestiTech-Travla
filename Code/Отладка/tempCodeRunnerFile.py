import cv2
import numpy as np
import os
import gradio as gr

# ==========================================
# ⚙️ БЛОК НАСТРОЕК (для веб-интерфейса)
# ==========================================
OUTPUT_DIR_ROOT = 'debugging_inspection'
os.makedirs(OUTPUT_DIR_ROOT, exist_ok=True)

def _order_corners(pts):
    """Упорядочивает 4 точки в порядке: верх-лево, верх-право, низ-право, низ-лево."""
    pts = pts.reshape(4, 2).astype(np.float32)
    ordered = np.zeros((4, 2), dtype=np.float32)

    s = pts.sum(axis=1)
    ordered[0] = pts[np.argmin(s)]   # верх-лево (min x+y)
    ordered[2] = pts[np.argmax(s)]   # низ-право (max x+y)

    diff = np.diff(pts, axis=1).ravel()
    ordered[1] = pts[np.argmin(diff)]  # верх-право (min y-x)
    ordered[3] = pts[np.argmax(diff)]  # низ-лево  (max y-x)
    return ordered


def _detect_board_quad(img_pcb, gerber_aspect):
    """
    Находит 4 угла платы на фото через GrabCut-сегментацию платы от фона (стола).

    GrabCut строит цветовые модели переднего/заднего плана, поэтому надёжно
    отделяет плату независимо от её цвета (зелёный текстолит или медь) и цвета
    стола — там, где обычный порог по яркости/цвету не справляется.

    Возвращает углы (TL, TR, BR, BL) либо None, если детекция не прошла проверки
    (тогда вызывающий код берёт полный кадр и ровную плату не обрезает).
    """
    h, w = img_pcb.shape[:2]

    # Работаем на уменьшенной копии — GrabCut дорогой, а край платы крупный.
    scale = 450.0 / max(h, w)
    small = cv2.resize(img_pcb, (max(1, int(w * scale)), max(1, int(h * scale))))
    sh, sw = small.shape[:2]

    # Инициализируем GrabCut прямоугольником, отступив от краёв: считаем, что
    # тонкая рамка по периметру кадра — это фон (стол).
    mask = np.zeros((sh, sw), np.uint8)
    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)
    mx, my = int(0.06 * sw), int(0.06 * sh)
    rect = (mx, my, sw - 2 * mx, sh - 2 * my)
    try:
        cv2.grabCut(small, mask, rect, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)
    except cv2.error:
        return None

    fg = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, kernel, iterations=3)
    fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, kernel, iterations=2)

    contours, _ = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    board = max(contours, key=cv2.contourArea)
    area_frac = cv2.contourArea(board) / float(sh * sw)

    # Минимальный повёрнутый прямоугольник платы (в координатах оригинала).
    rect_min = cv2.minAreaRect((board.astype(np.float32) / scale))
    (rw, rh) = rect_min[1]
    if rw < 1 or rh < 1:
        return None
    aspect = min(rw, rh) / max(rw, rh)

    # Проверка 1: плата должна занимать заметную часть кадра и совпадать
    # по пропорции сторон с гербером (иначе это ложная сегментация).
    if area_frac < 0.25 or abs(aspect - gerber_aspect) > 0.15:
        return None

    box = cv2.boxPoints(rect_min)

    # Проверка 2: не «прорезали» ли мы насквозь ровную плату, заполняющую кадр.
    # Сравниваем материал в тонком кольце снаружи бокса и внутри него: если он
    # одинаковый — значит плата продолжается за краем бокса (это не её край),
    # и обрезать нельзя.
    lab = cv2.cvtColor(img_pcb, cv2.COLOR_BGR2LAB)
    rmask = np.zeros((h, w), np.uint8)
    cv2.fillPoly(rmask, [box.astype(np.int32)], 255)
    d = max(3, int(0.03 * max(h, w)))
    ring_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (d, d))
    inner = cv2.subtract(rmask, cv2.erode(rmask, ring_k))
    outer = cv2.subtract(cv2.dilate(rmask, ring_k), rmask)
    if cv2.countNonZero(inner) == 0 or cv2.countNonZero(outer) == 0:
        return None
    color_inside = lab[inner > 0].mean(axis=0)
    color_outside = lab[outer > 0].mean(axis=0)
    if np.linalg.norm(color_inside - color_outside) < 10.0:
        return None  # снаружи ≈ внутри → плата заполняет кадр, край не найден

    return _order_corners(box)


def align_images_orb(img_gerber, img_pcb):
    """
    Выравнивание платы по герберу через детекцию края самой платы и
    перспективное преобразование по её 4 углам.

    В отличие от ORB (сопоставление ключевых точек + гомография с 8 степенями
    свободы), ровная плата не «выворачивается» в чужой внутренний контур:
    границы берутся строго по физическому краю платы. Перекошенные фото
    (например, PCB6, PCB_RESIST) выпрямляются по краю, а уже ровная плата,
    заполняющая кадр (PCB.jpg), остаётся нетронутой — используется полный кадр.
    """
    gh, gw = img_gerber.shape[:2]
    gerber_aspect = min(gw / gh, gh / gw)

    quad = _detect_board_quad(img_pcb, gerber_aspect)

    if quad is None:
        # Край платы не выделен (плата заполняет кадр или фон неотличим) —
        # берём полный кадр без искажения перспективы.
        ph, pw = img_pcb.shape[:2]
        quad = _order_corners(np.float32([[0, 0], [pw, 0], [pw, ph], [0, ph]]))

    # Целевой прямоугольник — размер гербера (сюда «распрямляется» плата).
    dst = np.float32([
        [0, 0],
        [gw - 1, 0],
        [gw - 1, gh - 1],
        [0, gh - 1],
    ])

    H = cv2.getPerspectiveTransform(quad, dst)
    img_pcb_aligned = cv2.warpPerspective(img_pcb, H, (gw, gh))

    return img_pcb_aligned


def _edge_map(img):
    """Карта краёв (Canny) — общий признак для гербера и фото, не зависит от цвета."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.createCLAHE(2.0, (8, 8)).apply(gray)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    edges = cv2.Canny(gray, 50, 150)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)
    return edges


def refine_registration_orb(img_gerber, img_pcb_aligned):
    """
    Тонкая доводка уже обрезанного изображения платы к герберу по ключевым точкам ORB.

    Сопоставление идёт по картам краёв (Canny), а не по цвету: границы дорожек
    находятся в одном месте независимо от полярности (медь на зелёном, синий резист
    на меди, белое на чёрном у гербера), поэтому работает для любых плат.

    Это КОРРЕКЦИЯ поверх обрезки, а не повторное совмещение: если опорных точек мало
    или найденное преобразование уводит углы кадра слишком далеко — доводка
    отбрасывается и возвращается исходное изображение (обрезка не страдает).
    """
    gh, gw = img_gerber.shape[:2]
    orb = cv2.ORB_create(nfeatures=6000)

    kg, dg = orb.detectAndCompute(_edge_map(img_gerber), None)
    kp, dp = orb.detectAndCompute(_edge_map(img_pcb_aligned), None)
    if dg is None or dp is None or len(kp) < 12 or len(kg) < 12:
        return img_pcb_aligned

    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    knn = bf.knnMatch(dp, dg, k=2)
    good = [m for m, n in (p for p in knn if len(p) == 2) if m.distance < 0.8 * n.distance]
    if len(good) < 30:
        return img_pcb_aligned

    src = np.float32([kp[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst = np.float32([kg[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    H, inliers = cv2.findHomography(src, dst, cv2.RANSAC, 4.0)
    if H is None or inliers is None or int(inliers.sum()) < 30:
        return img_pcb_aligned

    # Проверка адекватности: это доводка, поэтому углы кадра не должны
    # смещаться больше, чем на ~12% от размера. Иначе — отбрасываем.
    corners = np.float32([[0, 0], [gw, 0], [gw, gh], [0, gh]]).reshape(-1, 1, 2)
    moved = cv2.perspectiveTransform(corners, H).reshape(-1, 2)
    if np.linalg.norm(moved - corners.reshape(-1, 2), axis=1).max() > 0.12 * max(gw, gh):
        return img_pcb_aligned

    return cv2.warpPerspective(img_pcb_aligned, H, (gw, gh))


def binarize_pcb(img_aligned, block_size, c_val):
    """ Адаптивное выделение меди на основе локального контраста с ручными параметрами """
    b, g, r = cv2.split(img_aligned)
    
    # Медь сильнее отражает красный канал, подложка — зеленый
    diff = cv2.subtract(r, g)
    diff = cv2.normalize(diff, None, 0, 255, cv2.NORM_MINMAX)
    
    # Убедимся, что размер блока нечетный и больше 1
    if block_size % 2 == 0:
        block_size += 1
    if block_size < 3:
        block_size = 3

    # Бинаризация Гаусса с внешними параметрами
    thresh = cv2.adaptiveThreshold(
        diff, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY, block_size, c_val
    )
    
    # Убираем шумы
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    return thresh

def inspect_pcb_interface(gerber_img, pcb_img, block_size, c_val, min_area):
    """
    Основная функция-обработчик для Gradio интерфейса.
    Принимает изображения и настраиваемые параметры бинаризации.
    """
    if gerber_img is None or pcb_img is None:
        return None, None, "Пожалуйста, загрузите оба изображения (Шаблон и Фото платы)."

    # Конвертируем из RGB (Gradio) в BGR (OpenCV)
    img_gerber = cv2.cvtColor(gerber_img, cv2.COLOR_RGB2BGR)
    img_pcb = cv2.cvtColor(pcb_img, cv2.COLOR_RGB2BGR)

    status_msg = "Статус: Начинаем обработку...\n"
    
    try:
        # 1. Выравнивание: обрезка платы по её краю + тонкая доводка к герберу (ORB)
        img_pcb_aligned = align_images_orb(img_gerber, img_pcb)
        img_pcb_aligned = refine_registration_orb(img_gerber, img_pcb_aligned)
        cv2.imwrite(os.path.join(OUTPUT_DIR_ROOT, 'step1_pcb_aligned.jpg'), img_pcb_aligned)
        status_msg += "[1/3] Выравнивание слоев успешно завершено.\n"
        
        # 2. Бинаризация (с использованием переданных ползунков)
        gerber_gray = cv2.cvtColor(img_gerber, cv2.COLOR_BGR2GRAY)
        _, gerber_bin = cv2.threshold(gerber_gray, 127, 255, cv2.THRESH_BINARY)
        
        pcb_bin = binarize_pcb(img_pcb_aligned, int(block_size), int(c_val))
        cv2.imwrite(os.path.join(OUTPUT_DIR_ROOT, 'step2_pcb_binarized.jpg'), pcb_bin)
        status_msg += "[2/3] Сегментация меди завершена на основе выбранных параметров.\n"

        # Наложение маски
        kernel_roi = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
        roi_mask = cv2.dilate(gerber_bin, kernel_roi)

        gerber_active = cv2.bitwise_and(gerber_bin, roi_mask)
        pcb_active = cv2.bitwise_and(pcb_bin, roi_mask)

        # 3. Поиск дефектов
        kernel_clean = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        
        # Обрывы
        missing_copper = cv2.bitwise_and(gerber_active, cv2.bitwise_not(pcb_active))
        missing_copper = cv2.morphologyEx(missing_copper, cv2.MORPH_OPEN, kernel_clean)

        # Лишняя медь
        excess_copper = cv2.bitwise_and(pcb_active, cv2.bitwise_not(gerber_active))
        excess_copper = cv2.morphologyEx(excess_copper, cv2.MORPH_OPEN, kernel_clean)

        output_visual = img_pcb_aligned.copy()

        # Отрисовка обрывов (Красный)
        contours_missing, _ = cv2.findContours(missing_copper, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        breaks_count = 0
        for c in contours_missing:
            if cv2.contourArea(c) > min_area: 
                breaks_count += 1
                x, y, wc, hc = cv2.boundingRect(c)
                cv2.rectangle(output_visual, (x - 2, y - 2), (x + wc + 2, y + hc + 2), (0, 0, 255), 2)
                cv2.putText(output_visual, f"Break #{breaks_count}", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

        # Отрисовка лишней меди (Синий)
        contours_excess, _ = cv2.findContours(excess_copper, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        extras_count = 0
        for c in contours_excess:
            if cv2.contourArea(c) > min_area+10:
                extras_count += 1
                x, y, wc, hc = cv2.boundingRect(c)
                cv2.rectangle(output_visual, (x - 2, y - 2), (x + wc + 2, y + hc + 2), (255, 0, 0), 2)
                cv2.putText(output_visual, f"Extra #{extras_count}", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)

        # Сохраняем итоговый результат
        result_path = os.path.join(OUTPUT_DIR_ROOT, 'PCB_INSPECTION_RESULT.jpg')
        cv2.imwrite(result_path, output_visual)
        
        status_msg += f"[3/3] Анализ завершен!\nНайдено обрывов (Break): {breaks_count}\nНайдено излишков (Extra): {extras_count}\n\nРезультат сохранен в '{result_path}'"
        
        # Конвертируем изображения обратно в RGB для корректного отображения в Gradio
        output_visual_rgb = cv2.cvtColor(output_visual, cv2.COLOR_BGR2RGB)
        pcb_bin_rgb = cv2.cvtColor(pcb_bin, cv2.COLOR_GRAY2RGB)
        
        return output_visual_rgb, pcb_bin_rgb, status_msg

    except Exception as e:
        error_msg = f"[ОШИБКА ОБРАБОТКИ]: {str(e)}"
        print(error_msg)
        return None, None, error_msg

# ==========================================
# 🖥️ ОПИСАНИЕ ИНТЕРФЕЙСА GRADIO
# ==========================================
theme = gr.themes.Soft(primary_hue="blue", secondary_hue="gray")

with gr.Blocks(theme=theme, title="PCB Inspection AI") as demo:
    gr.Markdown(
        """
        # 🔍 Система автоматического контроля дефектов печатных плат
        Загрузите эталонное изображение (Gerber) и реальное фото вашей платы. Настройте ползунки бинаризации справа, чтобы получить идеальное выделение медных дорожек, убрав лишние шумы и текстолит.
        """
    )
    
    with gr.Row():
        # Левая колонка: Загрузка файлов
        with gr.Column(scale=1, min_width=300):
            gr.Markdown("### 📥 Шаг 1: Входные данные")
            gerber_input = gr.Image(label="Эталон (GERBER / Шаблон)", type="numpy")
            pcb_input = gr.Image(label="Фото платы (PCB)", type="numpy")
            
        # Средняя колонка: Интерактивные настройки
        with gr.Column(scale=1, min_width=300):
            gr.Markdown("### ⚙️ Шаг 2: Тонкая настройка бинаризации")
            
            # Слайдер для размера блока локальной адаптации (должен быть нечетным)
            block_size_slider = gr.Slider(
                minimum=3, maximum=101, step=2, value=51, 
                label="Размер локального блока (Adaptive Block Size)", 
                info="Большие значения сглаживают перепады освещения, меньшие выделяют мелкие детали."
            )
            
            # Слайдер для константы C
            c_val_slider = gr.Slider(
                minimum=-30, maximum=30, step=1, value=-15, 
                label="Смещение порога (Constant C)", 
                info="Уменьшайте (в минус), если на плате проступает текстура подложки. Увеличивайте, если пропадают дорожки."
            )
            
            # Минимальный размер дефекта
            min_area_slider = gr.Slider(
                minimum=10, maximum=1000, step=10, value=120, 
                label="Фильтр шума дефектов (Min Area)", 
                info="Игнорировать дефекты, площадь которых меньше указанного количества пикселей."
            )
            
            submit_btn = gr.Button("🚀 Запустить анализ", variant="primary")
            
        # Правая колонка: Вывод результатов
        with gr.Column(scale=1, min_width=300):
            gr.Markdown("### 📤 Шаг 3: Результаты")
            result_output = gr.Image(label="Найденные дефекты (Красный: Break / Синий: Extra)")
            bin_output = gr.Image(label="Ч/Б Маска меди (Результат бинаризации)")
            status_output = gr.Textbox(label="Лог работы системы", interactive=False, lines=5)

    # Связываем элементы интерфейса с обработчиком
    submit_btn.click(
        fn=inspect_pcb_interface,
        inputs=[
            gerber_input, 
            pcb_input, 
            block_size_slider, 
            c_val_slider, 
            min_area_slider
        ],
        outputs=[result_output, bin_output, status_output]
    )

if __name__ == "__main__":
    demo.launch(share=False)



