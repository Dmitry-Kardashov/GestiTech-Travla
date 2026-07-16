import cv2
import numpy as np
import os
import gradio as gr

# ==========================================
# ⚙️ БЛОК НАСТРОЕК
# ==========================================
OUTPUT_DIR_ROOT = 'debugging_inspection'
os.makedirs(OUTPUT_DIR_ROOT, exist_ok=True)

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
        return None

    return _order_corners(box)


def align_images_orb(img_gerber, img_pcb):
    gh, gw = img_gerber.shape[:2]
    gerber_aspect = min(gw / gh, gh / gw)

    quad = _detect_board_quad(img_pcb, gerber_aspect)

    if quad is None:
        ph, pw = img_pcb.shape[:2]
        quad = _order_corners(np.float32([[0, 0], [pw, 0], [pw, ph], [0, ph]]))

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
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.createCLAHE(2.0, (8, 8)).apply(gray)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    edges = cv2.Canny(gray, 50, 150)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)
    return edges


def refine_registration_orb(img_gerber, img_pcb_aligned):
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

    corners = np.float32([[0, 0], [gw, 0], [gw, gh], [0, gh]]).reshape(-1, 1, 2)
    moved = cv2.perspectiveTransform(corners, H).reshape(-1, 2)
    if np.linalg.norm(moved - corners.reshape(-1, 2), axis=1).max() > 0.12 * max(gw, gh):
        return img_pcb_aligned

    return cv2.warpPerspective(img_pcb_aligned, H, (gw, gh))


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



def binarize_pcb_raw(img_aligned, block_size, c_val):
    """ Шаг 2: Адаптивная бинаризация по цветовому каналу """
    b, g, r = cv2.split(img_aligned)
    diff = cv2.subtract(r, g)
    diff = cv2.normalize(diff, None, 0, 255, cv2.NORM_MINMAX)
    
    if block_size % 2 == 0:
        block_size += 1
    if block_size < 3:
        block_size = 3

    thresh_raw = cv2.adaptiveThreshold(
        diff, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY, block_size, c_val
    )
    return thresh_raw

def clean_pcb_mask(thresh_raw):
    """ Шаг 3: Морфологическое удаление шума """
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    thresh_clean = cv2.morphologyEx(thresh_raw, cv2.MORPH_OPEN, kernel)
    thresh_clean = cv2.morphologyEx(thresh_clean, cv2.MORPH_CLOSE, kernel)
    return thresh_clean

def process_pcb_pipeline(pcb_img, block_size, c_val):
    if pcb_img is None:
        return None, None, None, "Пожалуйста, загрузите изображение платы."

    img_pcb = cv2.cvtColor(pcb_img, cv2.COLOR_RGB2BGR)
    status_msg = "--- Старт обработки ---\n"
    
    try:
        # --- ШАГ 1: ОБРЕЗКА И ВЫРАВНИВАНИЕ ---
        img_pcb_aligned, align_status = align_and_crop_pcb(img_pcb)
        cv2.imwrite(os.path.join(OUTPUT_DIR_ROOT, 'step1_pcb_aligned.jpg'), img_pcb_aligned)
        status_msg += f"[Шаг 1]: {align_status}\n"
        
        # --- ШАГ 2: БИНАРИЗАЦИЯ ---
        pcb_bin_raw = binarize_pcb_raw(img_pcb_aligned, int(block_size), int(c_val))
        cv2.imwrite(os.path.join(OUTPUT_DIR_ROOT, 'step2_pcb_binarized_raw.jpg'), pcb_bin_raw)
        status_msg += "[Шаг 2]: Адаптивная бинаризация завершена.\n"

        # --- ШАГ 3: ОЧИСТКА ---
        pcb_bin_clean = clean_pcb_mask(pcb_bin_raw)
        cv2.imwrite(os.path.join(OUTPUT_DIR_ROOT, 'step3_pcb_binarized_clean.jpg'), pcb_bin_clean)
        status_msg += "[Шаг 3]: Фильтрация шумов завершена.\n"

        aligned_rgb = cv2.cvtColor(img_pcb_aligned, cv2.COLOR_BGR2RGB)
        bin_raw_rgb = cv2.cvtColor(pcb_bin_raw, cv2.COLOR_GRAY2RGB)
        bin_clean_rgb = cv2.cvtColor(pcb_bin_clean, cv2.COLOR_GRAY2RGB)
        
        return aligned_rgb, bin_raw_rgb, bin_clean_rgb, status_msg

    except Exception as e:
        error_msg = f"[ОШИБКА]: {str(e)}"
        return None, None, None, error_msg

# ==========================================
# 🖥️ ИНТЕРФЕЙС GRADIO
# ==========================================
theme = gr.themes.Soft(primary_hue="blue", secondary_hue="gray")

with gr.Blocks(theme=theme, title="PCB Step-by-Step Processing") as demo:
    gr.Markdown(
        """
        # 🔍 Пошаговая визуализация обработки печатных плат
        Программа автоматически находит границы платы, выпрямляет её, строит карту дорожек и убирает шумы.
        """
    )
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 📥 Входные данные")
            pcb_input = gr.Image(label="Фото платы (PCB)", type="numpy")
            
            gr.Markdown("### ⚙️ Параметры")
            block_size_slider = gr.Slider(
                minimum=3, maximum=151, step=2, value=51, 
                label="Размер блока бинаризации"
            )
            c_val_slider = gr.Slider(
                minimum=-30, maximum=30, step=1, value=-15, 
                label="Смещение порога (Constant C)"
            )
            
            submit_btn = gr.Button("🚀 Запустить обработку", variant="primary")
            status_output = gr.Textbox(label="Лог работы", interactive=False, lines=6)

        with gr.Column(scale=2):
            gr.Markdown("### 📤 Результаты по шагам")
            
            with gr.Tabs():
                with gr.TabItem("Шаг 1: Обрезанная плата"):
                    aligned_output = gr.Image(label="Обрезанное и выпрямленное фото")
                    
                with gr.TabItem("Шаг 2: Сырая маска"):
                    raw_bin_output = gr.Image(label="Результат бинаризации")
                    
                with gr.TabItem("Шаг 3: Очищенная маска"):
                    clean_bin_output = gr.Image(label="Результат без мелких шумов")

    submit_btn.click(
        fn=_order_corners,
        inputs=[pcb_input, block_size_slider, c_val_slider],
        outputs=[aligned_output, raw_bin_output, clean_bin_output, status_output]
    )

if __name__ == "__main__":
    demo.launch(share=False)