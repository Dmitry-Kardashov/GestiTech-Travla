import cv2
import numpy as np
import os
from pathlib import Path
from pygerber.gerberx3.api.v2 import GerberFile, ColorScheme, PixelFormatEnum
from pygerber.common.rgba import RGBA
import gradio as gr


import arduinoControl
import camera

css = """
.no-buttons [class*="button"], 
.no-buttons [id*="button"],
.no-buttons div[style*="position: absolute"] button,
.no-buttons svg {
    display: none !important;
    visibility: hidden !important;
    opacity: 0 !important;
    pointer-events: none !important;
}
"""


# ==========================================
# ⚙️ БЛОК НАСТРОЕК (КОНФИГУРАЦИЯ ПО УМОЛЧАНИЮ)
# ==========================================
DPMM = 40                      
TRACK_COLOR = "#FFFFFF"        
TRANSPARENT = True             

DEFAULT_CONFIG = {
    "output_dir": "debugging_inspection",      
    "path_output": "PCB_GBR.png",
    "max_working_side": 2200,
    "filter_d": 9,
    "sigma_color": 75,
    "sigma_space": 75,
    "block_size": 59,
    "c_val": -14,
    "noise_method": "Морфологическое открытие (Быстро)",
    "morph_size": 4,
    "min_noise_area": 250,
    "min_defect_area": 200,
    "large_defect_area": 200 * 4
}


# ==========================================
# 🛠️ ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ ОБРАБОТКИ
# ==========================================

def hex_to_rgba(hex_color: str, alpha: int = 255) -> RGBA:
    hex_color = hex_color.lstrip("#")
    r, g, b = (int(hex_color[i:i + 2], 16) for i in (0, 2, 4))
    return RGBA(r=r, g=g, b=b, a=alpha)

def gerber_to_png(input_path: str, output_path: str, dpmm: int, track_color: str, transparent_background: bool) -> None:
    color_scheme = ColorScheme(
        solid_color=hex_to_rgba(track_color, alpha=255),
        solid_region_color=hex_to_rgba(track_color, alpha=255),
        clear_color=hex_to_rgba(track_color, alpha=255),
        clear_region_color=hex_to_rgba(track_color, alpha=255),
        background_color=RGBA(r=0, g=0, b=0, a=0 if transparent_background else 255),
    )
    parsed = GerberFile.from_file(input_path).parse()
    parsed.render_raster(
        output_path,
        color_scheme=color_scheme,
        dpmm=dpmm,
        pixel_format=PixelFormatEnum.RGBA if transparent_background else PixelFormatEnum.RGB,
    )
    print(f"Готово: {output_path}")

def _resize_to_max_side(img, max_side):
    h, w = img.shape[:2]
    scale = max_side / float(max(h, w))
    if scale >= 1.0:
        return img
    new_w, new_h = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

def unify_resolution(img_gerber, img_pcb, max_working_side=None):
    gerber_side = max(img_gerber.shape[:2])
    pcb_side = max(img_pcb.shape[:2])
    target_side = min(gerber_side, pcb_side)
    if max_working_side:
        target_side = min(target_side, max_working_side)
    gerber_out = _resize_to_max_side(img_gerber, target_side)
    pcb_out = _resize_to_max_side(img_pcb, target_side)
    return gerber_out, pcb_out

def _order_corners(pts):
    pts = pts.reshape(4, 2).astype(np.float32)
    ordered = np.zeros((4, 2), dtype=np.float32)
    s = pts.sum(axis=1)
    ordered[0] = pts[np.argmin(s)]   
    ordered[2] = pts[np.argmax(s)]   
    diff = np.diff(pts, axis=1).ravel()
    ordered[1] = pts[np.argmin(diff)]  
    ordered[3] = pts[np.argmax(diff)]  
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
    dst = np.float32([[0, 0], [gw - 1, 0], [gw - 1, gh - 1], [0, gh - 1]])
    H = cv2.getPerspectiveTransform(quad, dst)
    return cv2.warpPerspective(img_pcb, H, (gw, gh))

def _edge_map(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.createCLAHE(2.0, (8, 8)).apply(gray)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    edges = cv2.Canny(gray, 50, 150)
    return cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)

def refine_registration_orb(img_gerber, img_pcb_aligned):
    gh, gw = img_gerber.shape[:2]
    sift = cv2.SIFT_create(nfeatures=10000)
    gray_gerber = cv2.cvtColor(img_gerber, cv2.COLOR_BGR2GRAY)
    gray_pcb = cv2.cvtColor(img_pcb_aligned, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
    gray_gerber = clahe.apply(gray_gerber)
    gray_pcb = clahe.apply(gray_pcb)
    kg, dg = sift.detectAndCompute(gray_gerber, None)
    kp, dp = sift.detectAndCompute(gray_pcb, None)
    
    if dg is None or dp is None or len(kp) < 10 or len(kg) < 10:
        return img_pcb_aligned
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(dp, dg, k=2)
    good = [m for m, n in matches if m.distance < 0.7 * n.distance]
    if len(good) < 15:
        return img_pcb_aligned
    src = np.float32([kp[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst = np.float32([kg[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    H, inliers = cv2.findHomography(src, dst, cv2.RANSAC, 6.0)
    if H is None or inliers is None or int(inliers.sum()) < 10:
        return img_pcb_aligned
    corners = np.float32([[0, 0], [gw, 0], [gw, gh], [0, gh]]).reshape(-1, 1, 2)
    moved = cv2.perspectiveTransform(corners, H).reshape(-1, 2)
    if np.linalg.norm(moved - corners.reshape(-1, 2), axis=1).max() > 0.3 * max(gw, gh):
        return img_pcb_aligned
    return cv2.warpPerspective(img_pcb_aligned, H, (gw, gh))

def verify_alignment(img_gerber, img_pcb_aligned, output_dir, min_match_ratio=0.5):
    edges_gerber = _edge_map(img_gerber)
    edges_pcb = _edge_map(img_pcb_aligned)
    edges_gerber_dilated = cv2.dilate(edges_gerber, np.ones((5, 5), np.uint8))
    total_pcb_edges = cv2.countNonZero(edges_pcb)
    matched_edges = cv2.countNonZero(cv2.bitwise_and(edges_pcb, edges_gerber_dilated))
    match_ratio = (matched_edges / total_pcb_edges) if total_pcb_edges > 0 else 0.0
    overlay = np.zeros((*edges_gerber.shape, 3), np.uint8)
    overlay[..., 2] = edges_gerber  
    overlay[..., 1] = edges_pcb     
    os.makedirs(output_dir, exist_ok=True)
    cv2.imwrite(os.path.join(output_dir, 'step2b_alignment_check.jpg'), overlay)
    return match_ratio

def remove_noise_by_contours(binary_img, min_area):
    if min_area <= 0:
        return binary_img
    contours, _ = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    clean_mask = np.zeros_like(binary_img)
    for cnt in contours:
        if cv2.contourArea(cnt) >= min_area:
            cv2.drawContours(clean_mask, [cnt], -1, 255, thickness=cv2.FILLED)
    return clean_mask

def binarize_pcb_advanced(img_aligned, filter_d, sigma_color, sigma_space, block_size, c_val, noise_method, morph_size, min_noise_area):
    b, g, r = cv2.split(img_aligned)
    diff = cv2.subtract(r, g)
    diff = cv2.normalize(diff, None, 0, 255, cv2.NORM_MINMAX)
    filtered = cv2.bilateralFilter(src=diff, d=int(filter_d), sigmaColor=sigma_color, sigmaSpace=sigma_space)
    otsu_val, _ = cv2.threshold(filtered, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    thr = int(np.clip(otsu_val - c_val, 0, 255))
    _, binary = cv2.threshold(filtered, thr, 255, cv2.THRESH_BINARY)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
    cleaned = binary.copy()
    if noise_method == "Морфологическое открытие (Быстро)":
        if morph_size > 0:
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (int(morph_size), int(morph_size)))
            cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel)
    elif noise_method == "Фильтрация по площади (Чисто)":
        cleaned = remove_noise_by_contours(cleaned, min_noise_area)
    return cleaned

def _local_topology(binary_img, x0, y0, x1, y1):
    crop = binary_img[y0:y1, x0:x1]
    contours, hierarchy = cv2.findContours(crop, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if hierarchy is None:
        return 0, 0
    hierarchy = hierarchy[0]
    n_components = sum(1 for h in hierarchy if h[3] == -1)   
    n_holes = sum(1 for h in hierarchy if h[3] != -1)        
    return n_components, n_holes

def _is_real_break(gerber_active, pcb_active, c, margin=6):
    h_img, w_img = gerber_active.shape
    x, y, w, h = cv2.boundingRect(c)
    x0, y0 = max(0, x - margin), max(0, y - margin)
    x1, y1 = min(w_img, x + w + margin), min(h_img, y + h + margin)
    ng_comp, ng_holes = _local_topology(gerber_active, x0, y0, x1, y1)
    np_comp, ng_holes_pcb = _local_topology(pcb_active, x0, y0, x1, y1)
    return (np_comp > ng_comp) or (ng_holes_pcb > ng_holes)

def _is_real_short(gerber_active, pcb_active, c, margin=6):
    h_img, w_img = gerber_active.shape
    x, y, w, h = cv2.boundingRect(c)
    x0, y0 = max(0, x - margin), max(0, y - margin)
    x1, y1 = min(w_img, x + w + margin), min(h_img, y + h + margin)
    ng_comp, _ = _local_topology(gerber_active, x0, y0, x1, y1)
    np_comp, _ = _local_topology(pcb_active, x0, y0, x1, y1)
    return (ng_comp > np_comp) or (ng_comp == 0 and np_comp > 0)

def smart_inspect_pcb(gerber_active, pcb_active, img_pcb_aligned, min_defect_area, large_defect_area):
    output_visual = img_pcb_aligned.copy()
    dist_gerber = cv2.distanceTransform(gerber_active, cv2.DIST_L2, 3)
    dist_pcb = cv2.distanceTransform(pcb_active, cv2.DIST_L2, 3)
    raw_missing = cv2.subtract(gerber_active, pcb_active)
    raw_excess = cv2.subtract(pcb_active, gerber_active)
    stats = {"critical_breaks": 0, "warnings_narrowing": 0, "critical_shorts": 0, "minor_excess": 0}

    contours_missing, _ = cv2.findContours(raw_missing, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in contours_missing:
        area = cv2.contourArea(c)
        if area < min_defect_area:
            continue
        mask_c = np.zeros_like(raw_missing)
        cv2.drawContours(mask_c, [c], -1, 255, -1)
        _, max_val_gerber, _, _ = cv2.minMaxLoc(dist_gerber, mask=mask_c)
        x, y, w, h = cv2.boundingRect(c)
        
        if area >= large_defect_area or _is_real_break(gerber_active, pcb_active, c) or max_val_gerber > 4.5:
            stats["critical_breaks"] += 1
            label = f"CRIT: Break #{stats['critical_breaks']}"
            color = (0, 0, 255)  
            thickness = 2
        else:
            stats["warnings_narrowing"] += 1
            label = f"WARN: Narrowing"
            color = (0, 165, 255)  
            thickness = 1
        cv2.rectangle(output_visual, (x - 3, y - 3), (x + w + 3, y + h + 3), color, thickness)
        cv2.putText(output_visual, label, (x, y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)

    contours_excess, _ = cv2.findContours(raw_excess, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    air_gerber = cv2.bitwise_not(gerber_active)
    dist_air = cv2.distanceTransform(air_gerber, cv2.DIST_L2, 3)

    for c in contours_excess:
        area = cv2.contourArea(c)
        if area < (min_defect_area + 10):
            continue
        mask_c = np.zeros_like(raw_excess)
        cv2.drawContours(mask_c, [c], -1, 255, -1)
        _, max_val_air, _, _ = cv2.minMaxLoc(dist_air, mask=mask_c)
        x, y, w, h = cv2.boundingRect(c)
        
        if area >= large_defect_area or _is_real_short(gerber_active, pcb_active, c) or max_val_air > 5.0:
            stats["critical_shorts"] += 1
            label = f"CRIT: Short #{stats['critical_shorts']}"
            color = (255, 0, 0)  
            thickness = 2
        else:
            stats["minor_excess"] += 1
            label = f"MINOR: Copper Splash"
            color = (255, 191, 0)  
            thickness = 1
        cv2.rectangle(output_visual, (x - 3, y - 3), (x + w + 3, y + h + 3), color, thickness)
        cv2.putText(output_visual, label, (x, y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)

    return output_visual, stats

# ==========================================
# 💻 ИНТЕГРАЦИЯ C GRADIO ВЕБ-ИНТЕРФЕЙСОМ
# ==========================================



def run_inspection(
    file_gbr, file_pcb_path, max_working_side, filter_d, 
    sigma_color, sigma_space, block_size, c_val, 
    noise_method, morph_size, min_noise_area, 
    min_defect_area, large_defect_area
):
    # Проверяем, выбраны ли файлы пользователем
    if file_gbr is None:
        return None, "Ошибка: Пожалуйста, выберите Gerber файл."
    if file_pcb_path is None:
        return None, "Ошибка: Пожалуйста, выберите фотографию платы."
    
    # Gradio сохраняет временные файлы. Получаем их абсолютные пути:
    path_gbr = file_gbr.name
    path_pcb = file_pcb_path
        
    os.makedirs(DEFAULT_CONFIG["output_dir"], exist_ok=True)
    
    # Шаг 1: Конвертация Gerber в PNG
    try:
        gerber_to_png(
            input_path=path_gbr,
            output_path=DEFAULT_CONFIG["path_output"],
            dpmm=DPMM,
            track_color=TRACK_COLOR,
            transparent_background=TRANSPARENT
        )
    except Exception as e:
        return None, f"Ошибка при обработке Gerber: {str(e)}"

    # Шаг 2: Чтение изображений
    img_gerber = cv2.imread(DEFAULT_CONFIG["path_output"])
    img_pcb = cv2.imread(path_pcb)
    
    # Шаг 3: Выравнивание разрешения
    img_gerber, img_pcb = unify_resolution(img_gerber, img_pcb, max_working_side)
    
    # Шаг 4: Регистрация / Выравнивание кадров
    img_pcb_aligned_rough = align_images_orb(img_gerber, img_pcb)
    img_pcb_aligned = refine_registration_orb(img_gerber, img_pcb_aligned_rough)
    
    # Проверка выравнивания
    verify_alignment(img_gerber, img_pcb_aligned, DEFAULT_CONFIG["output_dir"])
    
    # Шаг 5: Бинаризация
    gerber_gray = cv2.cvtColor(img_gerber, cv2.COLOR_BGR2GRAY)
    _, gerber_bin = cv2.threshold(gerber_gray, 127, 255, cv2.THRESH_BINARY)
    
    pcb_bin = binarize_pcb_advanced(
        img_pcb_aligned, filter_d, sigma_color, sigma_space, 
        block_size, c_val, noise_method, morph_size, min_noise_area
    )
    
    # Шаг 6: ROI маска
    kernel_roi = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    roi_mask = cv2.dilate(gerber_bin, kernel_roi)
    gerber_active = cv2.bitwise_and(gerber_bin, roi_mask)
    pcb_active = cv2.bitwise_and(pcb_bin, roi_mask)
    
    # Шаг 7: Поиск дефектов
    output_visual, stats = smart_inspect_pcb(
        gerber_active, pcb_active, img_pcb_aligned, min_defect_area, large_defect_area
    )
    
    # Переводим BGR в RGB для корректного отображения в Gradio
    output_visual_rgb = cv2.cvtColor(output_visual, cv2.COLOR_BGR2RGB)
    
    report_text = (
        f"--- ОБРАБОТКА ЗАВЕРШЕНА ---\n"
        f"Критических обрывов: {stats['critical_breaks']}\n"
        f"Предупреждений (сужения): {stats['warnings_narrowing']}\n"
        f"Критических замыканий: {stats['critical_shorts']}\n"
        f"Незначительных наплывов меди: {stats['minor_excess']}"
    )
    
    return output_visual_rgb, report_text



# Строим интерфейс Gradio
with gr.Blocks(title="Травилка") as demo:
    with gr.Row():
        # Первый логотип
        gr.Markdown("# Система определения печатных плат при травлении")
        gr.Image(
            value="Web/Логотип_black.svg", # Укажите реальный путь к первому логотипу
            show_label=False, 
            container=False, 
            height=80, 
            interactive=False, 
            elem_classes="no-buttons",
            # show_download_button=False
            buttons=[]  # Скрывает все встроенные кнопки
        )
        # Второй логотип (если нужен, или можно удалить этот gr.Image)
        gr.Image(
            value="Web/БВ Лого.svg", # Укажите реальный путь ко второму логотипу
            show_label=False, 
            container=False, 
            height=80, 
            interactive=False, 
            elem_classes="no-buttons",
            # show_download_button=False
            buttons=[]  # Скрывает все встроенные кнопки
        )
        # gr.HTML(f'<img src="file/Web/Логотип_black.svg" style="max-width:100%; height:80; border-radius:8px;">')


 

    with gr.Tabs():
        # Первая вкладка: Главная панель управления
        with gr.TabItem("Главная"):
            with gr.Row():
                with gr.Column():
                    gr.Markdown("### Выберите файлы в проводнике")
                    
                    # Заменили текстовые поля на кнопки загрузки файлов
                    input_gbr_file = gr.File(
                        label="Загрузить Gerber файл (.gbr)", 
                        file_count="single"
                    )
                    input_pcb_file = gr.Image(
                        label="Загрузить фото платы", 
                        type="filepath" # Передает путь к временному файлу
                    )
                    
                    with gr.Row():
                        btn_start_work = gr.Button("Начать работу", variant="secondary")
                        btn_run_main = gr.Button("Работа (тестирование)", variant="primary")
                        btn_open_cam = gr.Button("Открыть камеру", variant="secondary")
                    status_output = gr.Textbox(label="Статус системы / Лог пустой функции", placeholder="Здесь будет лог...")
                    
                with gr.Column():
                    gr.Markdown("### Результат анализа")
                    result_image = gr.Image(label="Карта дефектов")
                    result_report = gr.Textbox(label="Отчет", lines=6)

        # Вторая вкладка: Настройки
        with gr.TabItem("Настройки"):
            gr.Markdown("### Тонкая настройка алгоритмов обработки изображений")
            
            with gr.Row():
                with gr.Column():
                    gr.Markdown("#### Разрешение и геометрия")
                    cfg_max_working_side = gr.Number(label="max_working_side", value=DEFAULT_CONFIG["max_working_side"], precision=0)
                    
                    gr.Markdown("#### Билинейная фильтрация (Bilateral Filter)")
                    cfg_filter_d = gr.Slider(label="filter_d", minimum=1, maximum=25, step=1, value=DEFAULT_CONFIG["filter_d"])
                    cfg_sigma_color = gr.Slider(label="sigma_color", minimum=10, maximum=200, step=5, value=DEFAULT_CONFIG["sigma_color"])
                    cfg_sigma_space = gr.Slider(label="sigma_space", minimum=10, maximum=200, step=5, value=DEFAULT_CONFIG["sigma_space"])

                with gr.Column():
                    gr.Markdown("#### Бинаризация и Очистка шума")
                    cfg_block_size = gr.Slider(label="block_size", minimum=3, maximum=151, step=2, value=DEFAULT_CONFIG["block_size"])
                    cfg_c_val = gr.Slider(label="c_val", minimum=-50, maximum=50, step=1, value=DEFAULT_CONFIG["c_val"])
                    cfg_noise_method = gr.Dropdown(
                        label="noise_method", 
                        choices=["Без очистки", "Морфологическое открытие (Быстро)", "Фильтрация по площади (Чисто)"], 
                        value=DEFAULT_CONFIG["noise_method"]
                    )
                    cfg_morph_size = gr.Slider(label="morph_size", minimum=1, maximum=15, step=1, value=DEFAULT_CONFIG["morph_size"])
                    cfg_min_noise_area = gr.Number(label="min_noise_area", value=DEFAULT_CONFIG["min_noise_area"], precision=0)

                with gr.Column():
                    gr.Markdown("#### Порог площади дефектов")
                    cfg_min_defect_area = gr.Number(label="min_defect_area", value=DEFAULT_CONFIG["min_defect_area"], precision=0)
                    cfg_large_defect_area = gr.Number(label="large_defect_area (Критический порог)", value=DEFAULT_CONFIG["large_defect_area"], precision=0)

    # Логика кнопок
    btn_start_work.click(
        fn=arduinoControl.Arduino_Control,
        inputs=[],
        outputs=[status_output]
    )

    btn_open_cam.click(
        fn=camera.CameraInit,
        inputs=[],
        outputs=[]
    )
    
    btn_run_main.click(
        fn=run_inspection,
        inputs=[
            input_gbr_file, input_pcb_file, cfg_max_working_side, cfg_filter_d,
            cfg_sigma_color, cfg_sigma_space, cfg_block_size, cfg_c_val,
            cfg_noise_method, cfg_morph_size, cfg_min_noise_area,
            cfg_min_defect_area, cfg_large_defect_area
        ],
        outputs=[result_image, result_report]
    )

def main():
    demo.launch(share=False)
    # camera.CameraInit()


if __name__ == "__main__":
    main()


                       