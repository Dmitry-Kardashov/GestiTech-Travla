import os
import cv2
import glob
import numpy as np
from pathlib import Path
from pygerber.gerberx3.api.v2 import GerberFile, ColorScheme, PixelFormatEnum
from pygerber.common.rgba import RGBA

# Глобальные конфигурации по умолчанию
# DPMM снижен с 40 до 12: раньше Gerber рендерился в 8000x6000 (48 Мп) и тут же
# ужимался до max_working_side=2200 - гигантский рендер выбрасывался впустую и
# тормозил загрузку. Теперь рендерим сразу в рабочем разрешении (в разы быстрее,
# качество не теряется). Реальный dpmm ещё и подбирается адаптивно по размеру платы.
DPMM = 12
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

Input_skleika = 'pcb_pic'
output_skleika = 'PCB_Skleika.jpg'

def hex_to_rgba(hex_color: str, alpha: int = 255) -> RGBA:
    hex_color = hex_color.lstrip("#")
    r, g, b = (int(hex_color[i:i + 2], 16) for i in (0, 2, 4))
    return RGBA(r=r, g=g, b=b, a=alpha)

def _pick_dpmm(parsed, target_long_side, dpmm_default, dpmm_min=6, dpmm_max=40):
    """
    Подбирает dpmm так, чтобы длинная сторона отрендеренного PNG была ~target_long_side.
    Габариты платы (мм) берём из parsed.get_info(); если не вышло - dpmm_default.
    Смысл: рендерить сразу в рабочем разрешении, а не в 48 Мп с последующим ужатием.
    """
    try:
        info = parsed.get_info()
        long_side_mm = float(max(info.width_mm, info.height_mm))
        if long_side_mm > 0:
            dpmm = target_long_side / long_side_mm
            return int(max(dpmm_min, min(dpmm_max, round(dpmm))))
    except Exception as e:
        print(f"[gerber] Не удалось определить размер платы ({e}), dpmm={dpmm_default}.")
    return dpmm_default


def gerber_to_png(input_path: str, output_path: str, dpmm: int, track_color: str,
                  transparent_background: bool, target_long_side: int = None) -> None:
    color_scheme = ColorScheme(
        solid_color=hex_to_rgba(track_color, alpha=255),
        solid_region_color=hex_to_rgba(track_color, alpha=255),
        clear_color=hex_to_rgba(track_color, alpha=255),
        clear_region_color=hex_to_rgba(track_color, alpha=255),
        background_color=RGBA(r=0, g=0, b=0, a=0 if transparent_background else 255),
    )
    parsed = GerberFile.from_file(input_path).parse()
    if target_long_side:
        dpmm = _pick_dpmm(parsed, target_long_side, dpmm)
        print(f"[gerber] Рабочий dpmm={dpmm} (цель ~{target_long_side}px по длинной стороне).")
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

def _is_landscape(img) -> bool:
    h, w = img.shape[:2]
    return w >= h

def auto_rotate_to_match(img_gerber, img_pcb):
    """
    Снимок платы (особенно склеенная панорама) может оказаться повернут
    относительно Gerber-файла на 90/180/270 градусов - в зависимости от того,
    как плата была установлена и в какую сторону шла склейка кадров.
    Приводим img_pcb к той же ориентации (альбомная/книжная), что и img_gerber,
    а если это не снимает неоднозначность (90 vs 270, 0 vs 180 - у них
    одинаковый аспект), выбираем вариант с наибольшим числом ORB-совпадений
    с Gerber-файлом - перевернутый вариант почти всегда даст заметно меньше
    хороших совпадений.
    Возвращает (повернутое_изображение, угол_поворота_в_градусах).
    """
    gerber_landscape = _is_landscape(img_gerber)

    candidates = {
        0: img_pcb,
        90: cv2.rotate(img_pcb, cv2.ROTATE_90_CLOCKWISE),
        180: cv2.rotate(img_pcb, cv2.ROTATE_180),
        270: cv2.rotate(img_pcb, cv2.ROTATE_90_COUNTERCLOCKWISE),
    }

    # 1. Оставляем только повороты, дающие нужную (альбомную/книжную) ориентацию.
    same_orientation = [r for r in candidates if _is_landscape(candidates[r]) == gerber_landscape]
    if not same_orientation:
        same_orientation = list(candidates.keys())  # подстраховка, если аспекты совсем не совпадают

    if len(same_orientation) == 1:
        rot = same_orientation[0]
        return candidates[rot], rot

    # 2. Среди оставшихся кандидатов (обычно пара 0/180 или 90/270) выбираем
    #    по количеству хороших ORB-совпадений с Gerber-файлом.
    orb = cv2.ORB_create(2000)
    gray_gerber = cv2.cvtColor(img_gerber, cv2.COLOR_BGR2GRAY)
    kg, dg = orb.detectAndCompute(gray_gerber, None)

    best_rot, best_score = same_orientation[0], -1
    for rot in same_orientation:
        gray_pcb = cv2.cvtColor(candidates[rot], cv2.COLOR_BGR2GRAY)
        kp, dp = orb.detectAndCompute(gray_pcb, None)
        if dg is None or dp is None or len(kp) < 10 or len(kg) < 10:
            continue
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(dg, dp)
        good = [m for m in matches if m.distance < 50]
        score = len(good)
        if score > best_score:
            best_score = score
            best_rot = rot

    return candidates[best_rot], best_rot

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
    # nfeatures 10000 -> 3000: на рабочем разрешении ~2200px 3000 точек с запасом
    # хватает для RANSAC-гомографии, а SIFT в разы легче для Raspberry Pi 5.
    sift = cv2.SIFT_create(nfeatures=3000)
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

def _debug_save(img, filename: str, output_dir: str):
    """Сохраняет промежуточный кадр в папку отладки и пишет об этом в лог."""
    try:
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, filename)
        cv2.imwrite(path, img)
        print(f"[debug] Промежуточный кадр сохранен: {path}")
    except Exception as e:
        print(f"[debug] Не удалось сохранить промежуточный кадр {filename}: {e}")

def run_inspection(
    file_gbr, file_pcb_path, max_working_side, filter_d, 
    sigma_color, sigma_space, block_size, c_val, 
    noise_method, morph_size, min_noise_area, 
    min_defect_area, large_defect_area
):
    if file_gbr is not None:
        if hasattr(file_gbr, 'name'):
            gbr_path = file_gbr.name
        elif isinstance(file_gbr, str):
            gbr_path = file_gbr
        else:
            gbr_path = str(file_gbr)
    else:
        print("Ошибка: Gerber-файл не передан.")
        return None, "Ошибка: Gerber-файл не передан."
    
    if file_pcb_path is not None:
        if isinstance(file_pcb_path, str):
            path_pcb = file_pcb_path
        elif isinstance(file_pcb_path, dict) and 'name' in file_pcb_path:
            path_pcb = file_pcb_path['name']
        else:
            path_pcb = str(file_pcb_path)
    else:
        print("Ошибка: Фото платы не передано.")
        return None, "Ошибка: Фото платы не передано."
        
    os.makedirs(DEFAULT_CONFIG["output_dir"], exist_ok=True)
    debug_dir = DEFAULT_CONFIG["output_dir"]
    print(f"🔍 Запуск анализа. Gerber: {gbr_path}, Фото платы: {path_pcb}")
    print(f"[debug] Промежуточные кадры будут сохраняться в: {os.path.abspath(debug_dir)}")

    try:
        gerber_to_png(
            input_path=gbr_path,
            output_path=DEFAULT_CONFIG["path_output"],
            dpmm=DPMM,
            track_color=TRACK_COLOR,
            transparent_background=TRANSPARENT,
            target_long_side=int(max_working_side),
        )
    except Exception as e:
        print(f"Ошибка при обработке Gerber: {e}")
        return None, f"Ошибка при обработке Gerber: {str(e)}"

    img_gerber = cv2.imread(DEFAULT_CONFIG["path_output"])
    img_pcb = cv2.imread(path_pcb)
    
    if img_gerber is None or img_pcb is None:
        print("Ошибка чтения изображений с диска.")
        return None, "Ошибка чтения изображений с диска."

    _debug_save(img_gerber, "01_gerber_raw.jpg", debug_dir)
    _debug_save(img_pcb, "02_pcb_raw.jpg", debug_dir)

    # Снимок платы (особенно склеенная панорама) может прийти повернутым
    # на 90/180/270 градусов относительно Gerber-файла - приводим к нужной
    # ориентации до дальнейшего выравнивания.
    print("[debug] Проверка ориентации фото платы относительно Gerber-файла...")
    img_pcb, applied_rotation = auto_rotate_to_match(img_gerber, img_pcb)
    if applied_rotation != 0:
        print(f"Обнаружен поворот фото платы на {applied_rotation}° - скорректировано автоматически.")
        _debug_save(img_pcb, "03_pcb_rotated.jpg", debug_dir)
    else:
        print("[debug] Поворот фото платы не требуется.")

    print("[debug] Приведение изображений к единому разрешению...")
    img_gerber, img_pcb = unify_resolution(img_gerber, img_pcb, max_working_side)
    print(f"[debug] Рабочее разрешение: Gerber {img_gerber.shape[1]}x{img_gerber.shape[0]}, "
          f"PCB {img_pcb.shape[1]}x{img_pcb.shape[0]}")
    _debug_save(img_gerber, "04_gerber_resized.jpg", debug_dir)
    _debug_save(img_pcb, "05_pcb_resized.jpg", debug_dir)

    print("[debug] Грубое совмещение платы с Gerber-файлом (поиск контура платы)...")
    img_pcb_aligned_rough = align_images_orb(img_gerber, img_pcb)
    _debug_save(img_pcb_aligned_rough, "06_pcb_aligned_rough.jpg", debug_dir)

    print("[debug] Точное совмещение по ключевым точкам (SIFT/FLANN)...")
    img_pcb_aligned = refine_registration_orb(img_gerber, img_pcb_aligned_rough)
    _debug_save(img_pcb_aligned, "07_pcb_aligned_refined.jpg", debug_dir)

    match_ratio = verify_alignment(img_gerber, img_pcb_aligned, debug_dir)
    print(f"[debug] Качество совмещения (match_ratio): {match_ratio:.3f} "
          f"(см. также {os.path.join(debug_dir, 'step2b_alignment_check.jpg')})")

    print("[debug] Бинаризация Gerber-файла и фото платы...")
    gerber_gray = cv2.cvtColor(img_gerber, cv2.COLOR_BGR2GRAY)
    _, gerber_bin = cv2.threshold(gerber_gray, 127, 255, cv2.THRESH_BINARY)
    _debug_save(gerber_bin, "08_gerber_binary.jpg", debug_dir)

    pcb_bin = binarize_pcb_advanced(
        img_pcb_aligned, filter_d, sigma_color, sigma_space, 
        block_size, c_val, noise_method, morph_size, min_noise_area
    )
    _debug_save(pcb_bin, "09_pcb_binary.jpg", debug_dir)
    
    kernel_roi = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    roi_mask = cv2.dilate(gerber_bin, kernel_roi)
    gerber_active = cv2.bitwise_and(gerber_bin, roi_mask)
    pcb_active = cv2.bitwise_and(pcb_bin, roi_mask)
    _debug_save(gerber_active, "10_gerber_active.jpg", debug_dir)
    _debug_save(pcb_active, "11_pcb_active.jpg", debug_dir)

    print("[debug] Поиск дефектов (обрывы/замыкания)...")
    output_visual, stats = smart_inspect_pcb(
        gerber_active, pcb_active, img_pcb_aligned, min_defect_area, large_defect_area
    )
    _debug_save(output_visual, "12_output_visual.jpg", debug_dir)
    print(f"[debug] Найдено дефектов: {stats}")

    output_visual_rgb = cv2.cvtColor(output_visual, cv2.COLOR_BGR2RGB)
    
    report_text = (
        f"--- ОБРАБОТКА ЗАВЕРШЕНА ---\n"
        f"Критических обрывов: {stats['critical_breaks']}\n"
        f"Предупреждений (сужения): {stats['warnings_narrowing']}\n"
        f"Критических замыканий: {stats['critical_shorts']}\n"
        f"Незначительных наплывов меди: {stats['minor_excess']}"
    )
    print("✅ Анализ завершен.")
    
    return output_visual_rgb, report_text

# ============================ СКЛЕЙКА ПАНОРАМЫ ============================
# Плата снимается на просветном столе, пока мотор поднимает её равномерными
# шагами: камера неподвижна, кадры отличаются в основном ВЕРТИКАЛЬНЫМ сдвигом с
# большим перекрытием. Поэтому вместо полной перспективной гомографии (она
# накапливает ошибку и «заваливает» панораму) оцениваем частичное аффинное
# преобразование (сдвиг+поворот+масштаб) между соседними кадрами - это устойчиво
# и быстро (важно для Raspberry Pi 5).

# Переиспользуемые объекты (не создаём в цикле - экономия на Pi).
_STITCH_ORB = cv2.ORB_create(4000)
_STITCH_BF = cv2.BFMatcher(cv2.NORM_HAMMING)
_STITCH_CLAHE = cv2.createCLAHE(2.0, (8, 8))

# Минимум inliers, чтобы считать пару кадров надёжно совмещённой. Если меньше -
# скорее всего это граница между разными прогонами (плата «прыгнула») - обрываем.
_STITCH_MIN_INLIERS = 15


def _stitch_gray(img):
    return _STITCH_CLAHE.apply(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))


def _estimate_pair_transform(img_a, img_b):
    """
    Оценивает аффинное преобразование, переводящее точки img_b в систему img_a.
    Возвращает (M 2x3, число_inliers) или (None, 0).
    """
    ka, da = _STITCH_ORB.detectAndCompute(_stitch_gray(img_a), None)
    kb, db = _STITCH_ORB.detectAndCompute(_stitch_gray(img_b), None)
    if da is None or db is None or len(ka) < 4 or len(kb) < 4:
        return None, 0

    good = []
    for pair in _STITCH_BF.knnMatch(db, da, k=2):
        if len(pair) < 2:
            continue
        m, n = pair
        if m.distance < 0.75 * n.distance:
            good.append(m)
    if len(good) < 10:
        return None, 0

    src = np.float32([kb[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst = np.float32([ka[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    M, inliers = cv2.estimateAffinePartial2D(src, dst, method=cv2.RANSAC,
                                             ransacReprojThreshold=5.0)
    if M is None or inliers is None:
        return None, 0
    return M, int(inliers.sum())


def _autocrop_nonblack(img):
    """Обрезает пустые (чёрные) поля канваса по bounding box содержимого."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ys, xs = np.where(gray > 0)
    if len(xs) == 0:
        return img
    x0, x1 = xs.min(), xs.max() + 1
    y0, y1 = ys.min(), ys.max() + 1
    return img[y0:y1, x0:x1]


def stitch_all_from_folder(web_module_ref=None):
    extensions = ('*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG')
    image_paths = []
    for ext in extensions:
        image_paths.extend(glob.glob(os.path.join(Input_skleika, ext)))

    image_paths = sorted(image_paths)
    num_images = len(image_paths)
    if num_images < 2:
        print(f"Ошибка: В папке '{Input_skleika}' найдено картинок: {num_images}. Для склейки нужно минимум 2!")
        return False

    print(f"Успешно найдено {num_images} изображений в папке '{Input_skleika}'")
    images = [cv2.imread(path) for path in image_paths]

    # 1. Цепочка абсолютных аффинных трансформов от 1-го кадра.
    #    abs_M[i] переводит кадр i в систему координат кадра 0.
    abs_M = [np.eye(2, 3, dtype=np.float32)]
    for i in range(1, num_images):
        rel, ninl = _estimate_pair_transform(images[i - 1], images[i])
        if rel is None or ninl < _STITCH_MIN_INLIERS:
            # Ненадёжная пара - вероятно, граница между прогонами. Обрываем серию,
            # чтобы не «приклеивать» мусор (страховка: папка и так чистится перед прогоном).
            print(f"[склейка] Пара {i-1}->{i}: мало inliers ({ninl}) - останавливаю склейку "
                  f"на {i} кадрах (похоже на границу прогона).")
            break
        print(f"[склейка] Пара {i-1}->{i}: inliers={ninl}, dy={rel[1,2]:.0f}px")
        comp = np.vstack([abs_M[i - 1], [0, 0, 1]]) @ np.vstack([rel, [0, 0, 1]])
        abs_M.append(comp[:2, :].astype(np.float32))

    used = len(abs_M)
    images = images[:used]
    if used < 2:
        print("[склейка] Надёжно совместился только 1 кадр - панорама не построена.")
        return False

    # 2. Габариты канваса по углам всех кадров.
    all_pts = []
    for i, img in enumerate(images):
        h, w = img.shape[:2]
        corners = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
        all_pts.append(cv2.transform(corners, abs_M[i]))
    all_pts = np.concatenate(all_pts, axis=0)
    x_min, y_min = all_pts.min(axis=0).ravel()
    x_max, y_max = all_pts.max(axis=0).ravel()
    canvas_w = int(np.ceil(x_max - x_min))
    canvas_h = int(np.ceil(y_max - y_min))

    # 3. Композитинг с лёгким линейным пером в зоне перекрытия (сглаживает шов).
    result = np.zeros((canvas_h, canvas_w, 3), dtype=np.float32)
    weight = np.zeros((canvas_h, canvas_w), dtype=np.float32)
    for i, img in enumerate(images):
        M = abs_M[i].copy()
        M[0, 2] -= x_min
        M[1, 2] -= y_min
        warped = cv2.warpAffine(img, M, (canvas_w, canvas_h)).astype(np.float32)
        # Вес кадра: 1 внутри, плавно спадает к краям (feather по расстоянию до края).
        h, w = img.shape[:2]
        wmask = np.ones((h, w), np.float32)
        cv2.rectangle(wmask, (0, 0), (w - 1, h - 1), 0.0, 1)
        wmask = cv2.distanceTransform((wmask > 0).astype(np.uint8) * 255, cv2.DIST_L2, 3)
        wmask = np.clip(wmask / 40.0, 0.05, 1.0)
        wwarp = cv2.warpAffine(wmask, M, (canvas_w, canvas_h))
        result += warped * wwarp[..., None]
        weight += wwarp
    nz = weight > 1e-6
    result[nz] /= weight[nz][..., None]
    result = np.clip(result, 0, 255).astype(np.uint8)

    # 4. Обрезка пустых полей канваса.
    result = _autocrop_nonblack(result)

    output_dir = os.path.dirname(output_skleika)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    cv2.imwrite(output_skleika, result)
    print(f"Успех! Панорама {result.shape[1]}x{result.shape[0]} сохранена здесь: {output_skleika}")

    if web_module_ref and hasattr(web_module_ref, 'trigger_auto_inspection'):
        print("Панорама готова, вызываю web.trigger_auto_inspection() для автоматического поиска дефектов...")
        web_module_ref.trigger_auto_inspection()
    else:
        print("⚠️ web_module_ref не передан (или в нем нет trigger_auto_inspection) - "
              "автоматический поиск дефектов после склейки НЕ запущен.")

    return True