import cv2
import numpy as np
import os

# ==========================================
# ⚙️ БЛОК НАСТРОЕК (ПЕРЕМЕННЫЕ И КОНСТАНТЫ)
# ==========================================

INPUT_GERBER = 'GERBER.png'                         # Исходный эталонный шаблон
INPUT_PCB = 'PCB.jpg'                               # Фото вытравленной платы
OUTPUT_DIR_ROOT = 'debugging_inspection'            # Папка для результатов

# --- Параметры адаптивной бинаризации (для поиска меток) ---
ADAPTIVE_BLOCK_SIZE = 61  
ADAPTIVE_C = 10           

# --- Фильтрация контуров меток ---
MIN_CIRCULARITY = 0.4     
MAX_CIRCULARITY = 2.0     
MIN_AREA_COEFF = 0.01     
MAX_AREA_COEFF = 0.08     

# --- Параметры Хафа для меток ---
HOUGH_DP = 1.2                               
HOUGH_PARAM1 = 80                            
HOUGH_ACCUM_THRESHOLDS = (45, 35, 25)        
HOUGH_MIN_DIST_COEFF = 0.08                  
HOUGH_MIN_RADIUS_COEFF = 0.012               
HOUGH_MAX_RADIUS_COEFF = 0.04                

MERGE_MIN_DIST_COEFF = 0.02                  
BOARD_MIN_AREA_COEFF = 0.15                  
MORPH_KERNEL_BASE_COEFF = 0.02               
MORPH_KERNEL_MIN_SIZE = 15                   

CORNER_TOLERANCE_COEFF = 0.16                
SIDE_TOLERANCE_COEFF = 0.07                  
CORNER_NAMES = {"Верхний-Левый", "Верхний-Правый", "Нижний-Левый", "Нижний-Правый"}


# =====================================================================
# 📐 БЛОК АНАЛИЗА И ПОИСКА МЕТОК (ТВОЯ БАЗОВАЯ ЛОГИКА)
# =====================================================================
def detect_circles(img):
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    blur_kernel = int(w / 400)
    if blur_kernel % 2 == 0:
        blur_kernel += 1
    blur_kernel = max(3, blur_kernel)
    gray_blurred = cv2.medianBlur(gray, blur_kernel)

    pad = ADAPTIVE_BLOCK_SIZE
    gray_padded = cv2.copyMakeBorder(gray_blurred, pad, pad, pad, pad, cv2.BORDER_REFLECT101)
    thresh_p = cv2.adaptiveThreshold(
        gray_padded, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY_INV, ADAPTIVE_BLOCK_SIZE, ADAPTIVE_C
    )
    thresh = thresh_p[pad:pad + h, pad:pad + w]

    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    min_area = int((w * MIN_AREA_COEFF) ** 2)
    max_area = int((w * MAX_AREA_COEFF) ** 2)

    detected = []
    for c in contours:
        area = cv2.contourArea(c)
        perimeter = cv2.arcLength(c, True)
        if perimeter == 0:
            continue
        circularity = 4 * np.pi * area / (perimeter ** 2)
        if MIN_CIRCULARITY < circularity < MAX_CIRCULARITY and min_area < area < max_area:
            M = cv2.moments(c)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                radius = int(np.sqrt(area / np.pi))
                detected.append((cX, cY, radius))

    gray_for_hough = cv2.GaussianBlur(gray_blurred, (9, 9), 2)
    minR = int(w * HOUGH_MIN_RADIUS_COEFF)
    maxR = int(w * HOUGH_MAX_RADIUS_COEFF)
    min_hough_dist = int(w * HOUGH_MIN_DIST_COEFF)

    for accum_thresh in HOUGH_ACCUM_THRESHOLDS:
        circles = cv2.HoughCircles(
            gray_for_hough, cv2.HOUGH_GRADIENT, dp=HOUGH_DP, minDist=min_hough_dist,
            param1=HOUGH_PARAM1, param2=accum_thresh, minRadius=minR, maxRadius=maxR
        )
        if circles is not None:
            for c in circles[0]:
                detected.append((int(c[0]), int(c[1]), int(c[2])))

    merged = []
    min_dist = w * MERGE_MIN_DIST_COEFF
    for d in detected:
        if not any(np.hypot(d[0] - m[0], d[1] - m[1]) < min_dist for m in merged):
            merged.append(d)

    return merged

def detect_board_rect(img):
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    sat = hsv[:, :, 1]

    _, sat_mask = cv2.threshold(sat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _, dark_mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    combo = cv2.bitwise_or(sat_mask, dark_mask)

    k = max(MORPH_KERNEL_MIN_SIZE, int(w * MORPH_KERNEL_BASE_COEFF))
    kernel = np.ones((k, k), np.uint8)
    combo = cv2.morphologyEx(combo, cv2.MORPH_CLOSE, kernel)
    combo = cv2.morphologyEx(combo, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(combo, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    c = max(contours, key=cv2.contourArea)
    if cv2.contourArea(c) < BOARD_MIN_AREA_COEFF * w * h:
        return None

    rect = cv2.minAreaRect(c)
    return cv2.boxPoints(rect)

def rect_target_positions(corners):
    corners = np.array(corners, dtype=np.float64)
    center = corners.mean(axis=0)
    angles = np.arctan2(corners[:, 1] - center[1], corners[:, 0] - center[0])
    ordered = corners[np.argsort(angles)]
    
    start = np.argmin(ordered[:, 0] + ordered[:, 1])
    ordered = np.roll(ordered, -start, axis=0)

    names_corners = ["Верхний-Левый", "Верхний-Правый", "Нижний-Правый", "Нижний-Левый"]
    pts = {name: tuple(ordered[i]) for i, name in enumerate(names_corners)}
    pts["Верхний-Центр"] = tuple((ordered[0] + ordered[1]) / 2)
    pts["Правый-Центр"] = tuple((ordered[1] + ordered[2]) / 2)
    pts["Нижний-Центр"] = tuple((ordered[2] + ordered[3]) / 2)
    pts["Левый-Центр"] = tuple((ordered[3] + ordered[0]) / 2)
    return pts

def match_zones(target_positions, circles, diag):
    result = {}
    for name, (tx, ty) in target_positions.items():
        coeff = CORNER_TOLERANCE_COEFF if name in CORNER_NAMES else SIDE_TOLERANCE_COEFF
        tolerance = diag * coeff
        best, best_dist = None, None
        for (cx, cy, r) in circles:
            dist = np.hypot(cx - tx, cy - ty)
            if dist < tolerance and (best_dist is None or dist < best_dist):
                best, best_dist = (cx, cy, r), dist
        if best is not None:
            result[name] = best
    return result

def get_alignment_points(img):
    h, w = img.shape[:2]
    circles = detect_circles(img)
    board_corners = detect_board_rect(img)
    
    full_frame_corners = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float64)
    candidates = [("весь кадр", full_frame_corners)]
    if board_corners is not None:
        candidates.append(("контур платы", board_corners))

    best_matches = {}
    for label, corners in candidates:
        targets = rect_target_positions(corners)
        diag = np.hypot(*(np.max(corners, axis=0) - np.min(corners, axis=0)))
        matches = match_zones(targets, circles, diag)
        if len(matches) > len(best_matches):
            best_matches = matches
    return best_matches


# =====================================================================
# 🛠️ НОВЫЙ БЛОК: СЕГМЕНТАЦИЯ И КОНТРОЛЬ С ЗАНИЖЕННЫМ ПОРОГОМ
# =====================================================================

def binarize_pcb(img_aligned):
    """ Выделение меди на желто-зеленом текстолите через R-G разность. """
    b, g, r = cv2.split(img_aligned)
    
    # Извлекаем разность (в меди преобладает красный спектр, в текстолите — зеленый)
    diff = cv2.subtract(r, g)
    diff = cv2.normalize(diff, None, 0, 255, cv2.NORM_MINMAX)
    
    # Локальный порог Гаусса для вырезки проводников
    thresh = cv2.adaptiveThreshold(
        diff, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY, 51, -15
    )
    
    # Фильтруем мелкие шумы матрицы камеры
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    return thresh


def inspect_pcb(gerber_path, pcb_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    img_gerber = cv2.imread(gerber_path)
    img_pcb = cv2.imread(pcb_path)
    
    if img_gerber is None or img_pcb is None:
        print("[ОШИБКА] Не найдены входные изображения.")
        return

    print("[1/4] Сопоставление реперных точек...")
    matches_gerber = get_alignment_points(img_gerber)
    matches_pcb = get_alignment_points(img_pcb)

    common_keys = set(matches_gerber.keys()).intersection(set(matches_pcb.keys()))
    if len(common_keys) < 4:
        print(f"[ОШИБКА] Найдено мало общих реперных меток ({len(common_keys)}). Нужно >= 4.")
        return

    # Подготовка массивов точек для гомографии
    pts_gerber = np.array([[matches_gerber[k][0], matches_gerber[k][1]] for k in common_keys], dtype=np.float32)
    pts_pcb = np.array([[matches_pcb[k][0], matches_pcb[k][1]] for k in common_keys], dtype=np.float32)

    print("[2/4] Геометрическое выравнивание перспективы...")
    H, _ = cv2.findHomography(pts_pcb, pts_gerber, cv2.RANSAC, 5.0)
    gh, gw = img_gerber.shape[:2]
    img_pcb_aligned = cv2.warpPerspective(img_pcb, H, (gw, gh))
    cv2.imwrite(os.path.join(output_dir, 'step1_pcb_aligned.jpg'), img_pcb_aligned)

    print("[3/4] Поканальная бинаризация слоев меди...")
    gerber_gray = cv2.cvtColor(img_gerber, cv2.COLOR_BGR2GRAY)
    
    # На GERBER.png фон черный (0), дорожки белые (255)
    _, gerber_bin = cv2.threshold(gerber_gray, 127, 255, cv2.THRESH_BINARY)
    
    pcb_bin = binarize_pcb(img_pcb_aligned)
    cv2.imwrite(os.path.join(output_dir, 'step2_pcb_binarized.jpg'), pcb_bin)

    # --- СНИЖЕНИЕ ЧУВСТВИТЕЛЬНОСТИ ПО КРАЯМ ДОРОЖЕК ---
    # Увеличиваем буферную маску до 15x15 пикселей, чтобы прощать плате легкие микросдвиги
    kernel_roi = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    roi_mask = cv2.dilate(gerber_bin, kernel_roi)

    # Анализируем структуры исключительно внутри зоны трассировки
    gerber_active = cv2.bitwise_and(gerber_bin, roi_mask)
    pcb_active = cv2.bitwise_and(pcb_bin, roi_mask)

    print("[4/4] Локальный дифференциальный анализ дефектов...")
    # Очистка маски дефектов от мелкозернистого шума (4x4)
    kernel_clean = cv2.getStructuringElement(cv2.MORPH_RECT, (4, 4))
    
    # ОБРЫВЫ / ПЕРЕТРАВ (В Gerber есть, на плате нет)
    missing_copper = cv2.bitwise_and(gerber_active, cv2.bitwise_not(pcb_active))
    missing_copper = cv2.morphologyEx(missing_copper, cv2.MORPH_OPEN, kernel_clean)

    # ЛИШНЯЯ МЕДЬ / КРАТЕРЫ (На плате есть, в Gerber нет)
    excess_copper = cv2.bitwise_and(pcb_active, cv2.bitwise_not(gerber_active))
    excess_copper = cv2.morphologyEx(excess_copper, cv2.MORPH_OPEN, kernel_clean)

    # Копия для отрисовки дефектов
    output_visual = img_pcb_aligned.copy()

    # --- ЗАНИЖЕНИЕ ПОРОГА ПО ПЛОЩАДИ (ФИЛЬТР ШУМА И ГРЯЗИ) ---
    MIN_DEFECT_AREA = 50 # Игнорировать любые аномалии площадью меньше 50 пикселей

    # Поиск и маркировка ОБРЫВОВ (КРАСНЫЕ КВАДРАТЫ)
    contours_missing, _ = cv2.findContours(missing_copper, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in contours_missing:
        if cv2.contourArea(c) > MIN_DEFECT_AREA:  
            x, y, wc, hc = cv2.boundingRect(c)
            cv2.rectangle(output_visual, (x - 2, y - 2), (x + wc + 2, y + hc + 2), (0, 0, 255), 2)
            cv2.putText(output_visual, "Break", (x, y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)

    # Поиск и маркировка ЛИШНЕЙ МЕДИ (СИНИЕ КВАДРАТЫ)
    contours_excess, _ = cv2.findContours(excess_copper, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in contours_excess:
        if cv2.contourArea(c) > MIN_DEFECT_AREA:
            x, y, wc, hc = cv2.boundingRect(c)
            cv2.rectangle(output_visual, (x - 2, y - 2), (x + wc + 2, y + hc + 2), (255, 0, 0), 2)
            cv2.putText(output_visual, "Extra", (x, y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 0, 0), 1)

    cv2.imwrite(os.path.join(output_dir, 'PCB_INSPECTION_RESULT.jpg'), output_visual)
    print(f"\n[УСПЕХ] Контроль завершен. Грубые дефекты сохранены в: '{output_dir}/PCB_INSPECTION_RESULT.jpg'")


if __name__ == "__main__":
    inspect_pcb(INPUT_GERBER, INPUT_PCB, OUTPUT_DIR_ROOT)