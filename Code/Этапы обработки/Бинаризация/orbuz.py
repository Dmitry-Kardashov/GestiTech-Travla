import cv2
import numpy as np
import os
import json
import gradio as gr

OUTPUT_DIR_ROOT = 'debugging_inspection'
os.makedirs(OUTPUT_DIR_ROOT, exist_ok=True)

CONFIG_FILE = 'pcb_config.json'

def load_saved_config():
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"Ошибка чтения конфигурации: {e}")
    return {"comparison_mode": "Структурное сравнение краёв (Без бинаризации)", "min_defect_area": 80}

def save_current_config(mode, area):
    config = {"comparison_mode": mode, "min_defect_area": area}
    try:
        with open(CONFIG_FILE, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=4, ensure_ascii=False)
        return "✅ Параметры сохранены!"
    except Exception as e:
        return f"❌ Ошибка сохранения: {str(e)}"

# ==========================================
# 🛠️ ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ ВЫРАВНИВАНИЯ
# ==========================================

def _order_corners(pts):
    pts = pts.reshape(4, 2).astype(np.float32)
    ordered = np.zeros((4, 2), dtype=np.float32)
    s = pts.sum(axis=1)
    ordered[0] = pts[np.argmin(s)]   # верх-лево
    ordered[2] = pts[np.argmax(s)]   # низ-право
    diff = np.diff(pts, axis=1).ravel()
    ordered[1] = pts[np.argmin(diff)]  # верх-право
    ordered[3] = pts[np.argmax(diff)]  # низ-лево
    return ordered

def _detect_board_quad(img_pcb, gerber_aspect):
    h, w = img_pcb.shape[:2]
    scale = 450.0 / max(h, w)
    small = cv2.resize(img_pcb, (max(1, int(w * scale)), max(1, int(h * scale))))
    sh, sw = small.shape[:2]

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

    rect_min = cv2.minAreaRect((board.astype(np.float32) / scale))
    (rw, rh) = rect_min[1]
    if rw < 1 or rh < 1:
        return None
    aspect = min(rw, rh) / max(rw, rh)

    if area_frac < 0.25 or abs(aspect - gerber_aspect) > 0.15:
        return None

    box = cv2.boxPoints(rect_min)
    return _order_corners(box)

def align_images_orb(img_gerber, img_pcb):
    gh, gw = img_gerber.shape[:2]
    gerber_aspect = min(gw / gh, gh / gw)
    quad = _detect_board_quad(img_pcb, gerber_aspect)

    if quad is None:
        ph, pw = img_pcb.shape[:2]
        quad = _order_corners(np.float32([[0, 0], [pw, 0], [pw, ph], [0, ph]]))

    dst = np.float32([[0, 0], [gw - 1, 0], [gw - 1, gh - 1], [0, gh - 1]])
    H = cv2.getPerspectiveTransform(quad, dst)
    return cv2.warpPerspective(img_pcb, H, (gw, gh))

def _edge_map(img):
    """Выделяет чистые контуры (границы) на изображении."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.createCLAHE(2.0, (8, 8)).apply(gray)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    edges = cv2.Canny(gray, 40, 120)
    # Слегка утолщаем линии контуров для надежности сопоставления
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)
    return edges

def refine_registration_orb(img_gerber, img_pcb_aligned):
    gh, gw = img_gerber.shape[:2]
    orb = cv2.ORB_create(nfeatures=5000)

    kg, dg = orb.detectAndCompute(_edge_map(img_gerber), None)
    kp, dp = orb.detectAndCompute(_edge_map(img_pcb_aligned), None)
    if dg is None or dp is None or len(kp) < 12 or len(kg) < 12:
        return img_pcb_aligned

    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    knn = bf.knnMatch(dp, dg, k=2)
    good = [m for m, n in (p for p in knn if len(p) == 2) if m.distance < 0.8 * n.distance]
    if len(good) < 20:
        return img_pcb_aligned

    src = np.float32([kp[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst = np.float32([kg[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    H, inliers = cv2.findHomography(src, dst, cv2.RANSAC, 4.0)
    if H is None or inliers is None or int(inliers.sum()) < 20:
        return img_pcb_aligned

    return cv2.warpPerspective(img_pcb_aligned, H, (gw, gh))

# ==========================================
# 🔍 КОРНЕВАЯ ЛОГИКА АНАЛИЗА
# ==========================================

def inspect_pcb_interface(gerber_img, pcb_img, comparison_mode, min_defect_area):
    if gerber_img is None or pcb_img is None:
        return None, None, "Пожалуйста, загрузите оба изображения."

    img_gerber = cv2.cvtColor(gerber_img, cv2.COLOR_RGB2BGR)
    img_pcb = cv2.cvtColor(pcb_img, cv2.COLOR_RGB2BGR)
    status_msg = "Статус: Начинаем обработку...\n"
    
    try:
        # 1. Точнейшее совмещение по ключевым точкам (гомография)
        img_pcb_aligned = align_images_orb(img_gerber, img_pcb)
        img_pcb_aligned = refine_registration_orb(img_gerber, img_pcb_aligned)
        status_msg += "[1/3] Изображения выровнены пиксель-в-пиксель.\n"
        
        # Создаем маску Gerber для выделения рабочей зоны
        gerber_gray = cv2.cvtColor(img_gerber, cv2.COLOR_BGR2GRAY)
        _, gerber_bin = cv2.threshold(gerber_gray, 127, 255, cv2.THRESH_BINARY)
        kernel_roi = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
        roi_mask = cv2.dilate(gerber_bin, kernel_roi)

        if comparison_mode == "Бинаризация дорожек (Классический)":
            # Стандартная бинаризация платы
            gray_aligned = cv2.cvtColor(img_pcb_aligned, cv2.COLOR_BGR2GRAY)
            pcb_bin = cv2.adaptiveThreshold(
                gray_aligned, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 15, 4
            )
            gerber_active = cv2.bitwise_and(gerber_bin, roi_mask)
            pcb_active = cv2.bitwise_and(pcb_bin, roi_mask)

            # Логическая разность масок
            missing_copper = cv2.bitwise_and(gerber_active, cv2.bitwise_not(pcb_active))
            excess_copper = cv2.bitwise_and(pcb_active, cv2.bitwise_not(gerber_active))
            visual_mask = pcb_bin

        else:  # "Структурное сравнение краёв (Без бинаризации)"
            # Анализируем разности контуров (перепадов градиента)
            edges_gerber = _edge_map(img_gerber)
            edges_pcb = _edge_map(img_pcb_aligned)
            
            edges_gerber_active = cv2.bitwise_and(edges_gerber, roi_mask)
            edges_pcb_active = cv2.bitwise_and(edges_pcb, roi_mask)

            # Вычитание карт краев друг из друга
            missing_copper = cv2.subtract(edges_gerber_active, edges_pcb_active)
            excess_copper = cv2.subtract(edges_pcb_active, edges_gerber_active)
            visual_mask = edges_pcb

        # Легкая морфологическая фильтрация результатов разницы от шума
        kernel_clean = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        missing_copper = cv2.morphologyEx(missing_copper, cv2.MORPH_OPEN, kernel_clean)
        excess_copper = cv2.morphologyEx(excess_copper, cv2.MORPH_OPEN, kernel_clean)

        output_visual = img_pcb_aligned.copy()

        # Поиск контуров дефектов
        contours_missing, _ = cv2.findContours(missing_copper, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        breaks_count = 0
        for c in contours_missing:
            if cv2.contourArea(c) > min_defect_area: 
                breaks_count += 1
                x, y, wc, hc = cv2.boundingRect(c)
                cv2.rectangle(output_visual, (x - 2, y - 2), (x + wc + 2, y + hc + 2), (0, 0, 255), 2)
                cv2.putText(output_visual, f"Break #{breaks_count}", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

        contours_excess, _ = cv2.findContours(excess_copper, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        extras_count = 0
        for c in contours_excess:
            if cv2.contourArea(c) > min_defect_area:
                extras_count += 1
                x, y, wc, hc = cv2.boundingRect(c)
                cv2.rectangle(output_visual, (x - 2, y - 2), (x + wc + 2, y + hc + 2), (255, 0, 0), 2)
                cv2.putText(output_visual, f"Extra #{extras_count}", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)

        status_msg += f"[2/3] Анализ структуры окончен.\n[3/3] Результаты:\n- Недостающие элементы (Обрывы): {breaks_count}\n- Лишние элементы (КЗ/Грязь): {extras_count}"
        
        output_visual_rgb = cv2.cvtColor(output_visual, cv2.COLOR_BGR2RGB)
        visual_mask_rgb = cv2.cvtColor(visual_mask, cv2.COLOR_GRAY2RGB)
        
        return output_visual_rgb, visual_mask_rgb, status_msg

    except Exception as e:
        return None, None, f"[Ошибка]: {str(e)}"

# ==========================================
# 🖥️ ИНТЕРФЕЙС GRADIO
# ==========================================
theme = gr.themes.Soft(primary_hue="blue", secondary_hue="gray")
initial_cfg = load_saved_config()

with gr.Blocks(title="PCB Inspection AI") as demo:
    gr.Markdown("# 🔍 Контроль дефектов печатных плат")
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 📥 Входные данные")
            gerber_input = gr.Image(label="Эталон (Gerber / Чертеж)", type="numpy")
            pcb_input = gr.Image(label="Фото платы (PCB)", type="numpy")
            
        with gr.Column(scale=1):
            gr.Markdown("### ⚙️ Настройки анализа")
            
            comparison_mode_selector = gr.Dropdown(
                choices=["Структурное сравнение краёв (Без бинаризации)", "Бинаризация дорожек (Классический)"],
                value=initial_cfg.get("comparison_mode", "Структурное сравнение краёв (Без бинаризации)"),
                label="🎯 Метод сравнения",
                info="Выберите структурный метод для анализа без сложной бинаризации."
            )
            
            min_defect_area_slider = gr.Slider(
                minimum=10, maximum=1000, step=10, 
                value=initial_cfg.get("min_defect_area", 80), 
                label="Минимальный размер дефекта (в пикселях)", 
                info="Помогает отсеять мелкую пыль, шумы камеры и артефакты."
            )
            
            with gr.Row():
                submit_btn = gr.Button("🚀 Запустить анализ", variant="primary")
                save_btn = gr.Button("💾 Запомнить выбор", variant="secondary")
                
            save_status = gr.Markdown()
            
        with gr.Column(scale=1):
            gr.Markdown("### 📤 Результаты")
            result_output = gr.Image(label="Карта дефектов (Красный: Недостает / Синий: Лишнее)")
            bin_output = gr.Image(label="Промежуточный анализ (Карта краёв / Маска)")
            status_output = gr.Textbox(label="Лог системы", interactive=False, lines=4)

    # Клиентская логика
    save_btn.click(
        fn=save_current_config,
        inputs=[comparison_mode_selector, min_defect_area_slider],
        outputs=[save_status]
    )

    submit_btn.click(
        fn=inspect_pcb_interface,
        inputs=[gerber_input, pcb_input, comparison_mode_selector, min_defect_area_slider],
        outputs=[result_output, bin_output, status_output]
    )

    demo.load(
        fn=lambda: (initial_cfg.get("comparison_mode", "Структурное сравнение краёв (Без бинаризации)"), initial_cfg.get("min_defect_area", 80)),
        outputs=[comparison_mode_selector, min_defect_area_slider]
    )

if __name__ == "__main__":
    demo.launch(share=False, theme=theme)