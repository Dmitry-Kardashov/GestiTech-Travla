import cv2
import numpy as np
import os

# ==========================================
# ⚙️ БЛОК НАСТРОЕК (ПЕРЕМЕННЫЕ И КОНСТАНТЫ)
# ==========================================

# --- Настройки путей и файлов ---
# INPUT_IMAGES = ['PCB.jpg']  # Список изображений плат для обработки
INPUT_IMAGES = ['PCB6.png']  # Список изображений плат для обработки
# INPUT_IMAGES = ['GERBER.png']  # Список изображений плат для обработки
OUTPUT_DIR_ROOT = 'debugging2'                      # Корневая папка для сохранения отладочных кадров

# --- Параметры адаптивной бинаризации ---
ADAPTIVE_BLOCK_SIZE = 61  # Размер окрестности пикселя для расчета порога (должен быть нечетным)
ADAPTIVE_C = 10           # Константа, вычитаемая из среднего или взвешенного среднего

# --- Фильтрация контуров по форме и площади ---
MIN_CIRCULARITY = 0.4     # Минимальный коэффициент округлости контура
MAX_CIRCULARITY = 2.0     # Максимальный коэффициент округлости контура
MIN_AREA_COEFF = 0.01     # Коэффициент минимальной площади контура относительно ширины кадра (w * COEFF)
MAX_AREA_COEFF = 0.08     # Коэффициент максимальной площади контура относительно ширины кадра (w * COEFF)

# --- Параметры преобразования Хафа (HoughCircles) ---
HOUGH_DP = 1.2                               # Обратное соотношение разрешения накопителя к разрешению изображения
HOUGH_PARAM1 = 80                            # Выше порог — меньше ложных кругов (передается в Кэнни)
HOUGH_ACCUM_THRESHOLDS = (45, 35, 25)        # Пороги живучести центров (param2) для итеративного поиска
HOUGH_MIN_DIST_COEFF = 0.08                  # Мин. расстояние между центрами кругов относительно ширины кадра
HOUGH_MIN_RADIUS_COEFF = 0.012               # Минимальный радиус круга относительно ширины кадра
HOUGH_MAX_RADIUS_COEFF = 0.04                # Максимальный радиус круга относительно ширины кадра

# --- Удаление дубликатов меток ---
MERGE_MIN_DIST_COEFF = 0.02                  # Радиус склейки близких меток относительно ширины кадра

# --- Поиск контура платы ---
BOARD_MIN_AREA_COEFF = 0.15                  # Минимальная площадь платы относительно всей площади кадра (0.15 = 15%)
MORPH_KERNEL_BASE_COEFF = 0.02               # Коэффициент размера ядра морфологии относительно ширины кадра
MORPH_KERNEL_MIN_SIZE = 15                   # Минимальный размер ядра морфологии (в пикселях)

# --- Выравнивание платы (перспектива) ---
BOARD_MARGIN_COEFF = 0.06                    # Наружный отступ за угловые метки, чтобы в кадр попал край платы

# --- Сопоставление зон (Допуски) ---
CORNER_TOLERANCE_COEFF = 0.16                # Допуск поиска круга для угловых меток (относительно диагонали)
SIDE_TOLERANCE_COEFF = 0.07                  # Допуск поиска круга для боковых меток (относительно диагонали)

# --- Системные константы ---
CORNER_NAMES = {"Верхний-Левый", "Верхний-Правый", "Нижний-Левый", "Нижний-Правый"}
# ==========================================


# =====================================================================
# 📐 ПОИСК КРУГЛЫХ МЕТОК (ФИДУЦИАЛЬНЫХ ОТВЕРСТИЙ)
# =====================================================================
def detect_circles(img):
    """
    Ищет круглые метки двумя взаимно дополняющими способами и объединяет результаты.
    """
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Размер ядра размытия подбираем от ширины кадра
    blur_kernel = int(w / 400)
    if blur_kernel % 2 == 0:
        blur_kernel += 1
    blur_kernel = max(3, blur_kernel)
    gray_blurred = cv2.medianBlur(gray, blur_kernel)

    # --- Способ 1: адаптивная бинаризация + круглость контура ---
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

    # --- Способ 2: преобразование Хафа (несколько порогов чувствительности) ---
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

    # Убираем дубликаты — оба метода часто находят одну и ту же метку
    merged = []
    min_dist = w * MERGE_MIN_DIST_COEFF
    for d in detected:
        if not any(np.hypot(d[0] - m[0], d[1] - m[1]) < min_dist for m in merged):
            merged.append(d)

    return merged, thresh


# =====================================================================
# 📐 ПОИСК КОНТУРА САМОЙ ПЛАТЫ НА ФОТО
# =====================================================================
def detect_board_rect(img):
    """
    Пытается отделить плату от фона по насыщенности цвета и яркости,
    и вернуть повёрнутый прямоугольник платы.
    """
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


def order_corners(pts):
    """
    Упорядочивает 4 точки прямоугольника в порядке:
    Верхний-Левый, Верхний-Правый, Нижний-Правый, Нижний-Левый.
    Устойчиво к повороту и зеркальному отражению.
    """
    pts = np.array(pts, dtype=np.float32)
    s = pts.sum(axis=1)              # x + y
    diff = pts[:, 0] - pts[:, 1]     # x - y

    tl = pts[np.argmin(s)]       # минимальная сумма — верхний левый
    br = pts[np.argmax(s)]       # максимальная сумма — нижний правый
    tr = pts[np.argmax(diff)]    # максимальная разность (x-y) — верхний правый
    bl = pts[np.argmin(diff)]    # минимальная разность — нижний левый
    return np.array([tl, tr, br, bl], dtype=np.float32)


def estimate_board_corners(matches):
    """
    Восстанавливает 4 угла платы по найденным меткам.

    Метки образуют прямоугольную сетку: 4 угла и 4 середины сторон, где
    каждая середина = полусумма двух соседних углов. Поэтому недостающий угол
    можно восстановить из середины стороны и противоположного ей угла:
        Верхний-Левый = 2*Верхний-Центр - Верхний-Правый  и т.п.

    Возвращает массив [ВЛ, ВП, НП, НЛ] или None, если данных не хватает.
    """
    P = {name: np.array([v[0], v[1]], dtype=np.float64) for name, v in matches.items()}
    C = {n: P[n] for n in ("Верхний-Левый", "Верхний-Правый",
                           "Нижний-Правый", "Нижний-Левый") if n in P}

    # Для каждого угла — пары (середина стороны, соседний угол на той же стороне)
    rel = {
        "Верхний-Левый":  [("Верхний-Центр", "Верхний-Правый"), ("Левый-Центр", "Нижний-Левый")],
        "Верхний-Правый": [("Верхний-Центр", "Верхний-Левый"),  ("Правый-Центр", "Нижний-Правый")],
        "Нижний-Правый":  [("Правый-Центр", "Верхний-Правый"),  ("Нижний-Центр", "Нижний-Левый")],
        "Нижний-Левый":   [("Нижний-Центр", "Нижний-Правый"),   ("Левый-Центр", "Верхний-Левый")],
    }

    # Несколько проходов: восстановленный угол помогает восстановить следующий
    for _ in range(4):
        for corner, rules in rel.items():
            if corner in C:
                continue
            estimates = [2 * P[mid] - C[other]
                         for mid, other in rules if mid in P and other in C]
            if estimates:
                C[corner] = np.mean(estimates, axis=0)

    if len(C) < 4:
        return None
    return np.array([C["Верхний-Левый"], C["Верхний-Правый"],
                     C["Нижний-Правый"], C["Нижний-Левый"]], dtype=np.float64)


def expand_corners(corners, margin):
    """Раздвигает углы прямоугольника наружу от центра на долю margin."""
    corners = np.array(corners, dtype=np.float64)
    center = corners.mean(axis=0)
    return center + (corners - center) * (1.0 + margin)


def align_board(img, board_corners):
    """
    Выравнивает плату в строгий прямоугольник с помощью перспективного
    преобразования. Возвращает выпрямленное изображение и матрицу перехода.

    board_corners — 4 угла повёрнутого прямоугольника платы (cv2.boxPoints).
    """
    src = order_corners(board_corners)
    (tl, tr, br, bl) = src

    # Ширина/высота результата — по реальным длинам сторон платы
    width_top = np.hypot(*(tr - tl))
    width_bottom = np.hypot(*(br - bl))
    height_left = np.hypot(*(bl - tl))
    height_right = np.hypot(*(br - tr))

    dst_w = int(round(max(width_top, width_bottom)))
    dst_h = int(round(max(height_left, height_right)))
    if dst_w < 1 or dst_h < 1:
        return None, None

    dst = np.array([
        [0, 0],
        [dst_w - 1, 0],
        [dst_w - 1, dst_h - 1],
        [0, dst_h - 1],
    ], dtype=np.float32)

    M = cv2.getPerspectiveTransform(src, dst)
    aligned = cv2.warpPerspective(img, M, (dst_w, dst_h))
    return aligned, M


def rect_target_positions(corners):
    """
    Возвращает 8 именованных целевых точек: 4 угла + 4 середины сторон.
    """
    corners = np.array(corners, dtype=np.float64)
    center = corners.mean(axis=0)
    angles = np.arctan2(corners[:, 1] - center[1], corners[:, 0] - center[0])
    ordered = corners[np.argsort(angles)]
    
    # Начинаем обход с точки, ближайшей к верхнему левому углу кадра
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
    """
    Для каждой из 8 целевых позиций ищем ближайшую найденную окружность с учетом допусков.
    """
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


def process_image(input_path, output_dir_root):
    output_dir = os.path.join(output_dir_root, os.path.splitext(os.path.basename(input_path))[0])
    os.makedirs(output_dir, exist_ok=True)

    img = cv2.imread(input_path)
    if img is None:
        print(f"[ОШИБКА] Не удалось прочитать файл: {input_path}")
        return
    h, w = img.shape[:2]

    circles, thresh = detect_circles(img)
    cv2.imwrite(os.path.join(output_dir, 'step2_threshold.jpg'), thresh)

    img_all_valid_circles = img.copy()
    for (cx, cy, r) in circles:
        cv2.circle(img_all_valid_circles, (cx, cy), r, (255, 0, 0), 2)
    cv2.imwrite(os.path.join(output_dir, 'step4_detected_circles.jpg'), img_all_valid_circles)

    full_frame_corners = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float64)
    board_corners = detect_board_rect(img)

    candidates = [("весь кадр", full_frame_corners)]
    if board_corners is not None:
        candidates.append(("контур платы", board_corners))

    best_label, best_matches, best_targets = None, {}, None
    for label, corners in candidates:
        targets = rect_target_positions(corners)
        diag = np.hypot(*(np.max(corners, axis=0) - np.min(corners, axis=0)))
        matches = match_zones(targets, circles, diag)
        if len(matches) > len(best_matches):
            best_label, best_matches, best_targets = label, matches, targets

    # Отрисовка и вывод информации
    img_final = img.copy()
    print(f"\n--- Результаты распознавания меток: {input_path} ---")
    print(f"    (система координат платы определена как: {best_label})")
    for zone_name, target_pos in best_targets.items():
        tx, ty = int(target_pos[0]), int(target_pos[1])
        if zone_name in best_matches:
            cX, cY, radius = best_matches[zone_name]
            print(f"[ НАЙДЕН ] {zone_name:15} -> Координаты: ({cX}, {cY}), Радиус: {radius}")
            cv2.circle(img_final, (cX, cY), radius + 5, (0, 255, 0), 3)
            cv2.circle(img_final, (cX, cY), 3, (0, 0, 255), -1)
            cv2.putText(img_final, zone_name, (cX - 40, cY - radius - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        else:
            print(f"[ОТСУТСТВУЕТ] {zone_name:15}")
            cv2.drawMarker(img_final, (tx, ty), (0, 0, 255), cv2.MARKER_CROSS, 20, 2)

    cv2.imwrite(os.path.join(output_dir, 'step5_final_corners.jpg'), img_final)

    # =====================================================================
    # 📐 БЛОК ВЫРАВНИВАНИЯ ПЕРСПЕКТИВЫ
    # =====================================================================
    # Приоритет — углы, восстановленные по меткам: они привязаны к самой плате
    # и надёжнее, чем контур (который часто цепляется за скопление меди).
    # aligned = None
    # warp_matrix = None
    # align_source = None
    # align_kind = None

    # marks_corners = estimate_board_corners(best_matches)
    # if marks_corners is not None:
    #     # Раздвигаем наружу, чтобы физический край платы тоже попал в кадр
    #     align_source = expand_corners(marks_corners, BOARD_MARGIN_COEFF)
    #     align_kind = "по меткам"
    # elif board_corners is not None:
    #     align_source = board_corners
    #     align_kind = "по контуру"

    # if align_source is not None:
    #     aligned, warp_matrix = align_board(img, align_source)

    # if aligned is not None:
    #     # Перерисуем найденные метки на выровненном изображении
    #     aligned_annotated = aligned.copy()
    #     for zone_name, (cX, cY, radius) in best_matches.items():
    #         pt = np.array([[[float(cX), float(cY)]]], dtype=np.float32)
    #         wx, wy = cv2.perspectiveTransform(pt, warp_matrix)[0][0]
    #         wx, wy = int(round(wx)), int(round(wy))
    #         cv2.circle(aligned_annotated, (wx, wy), radius + 5, (0, 255, 0), 3)
    #         cv2.circle(aligned_annotated, (wx, wy), 3, (0, 0, 255), -1)
    #         cv2.putText(aligned_annotated, zone_name, (wx - 40, wy - radius - 10),
    #                     cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    #     cv2.imwrite(os.path.join(output_dir, 'step6_aligned.jpg'), aligned)
    #     cv2.imwrite(os.path.join(output_dir, 'step6_aligned_marked.jpg'), aligned_annotated)
    #     print(f"    (плата выровнена {align_kind} в прямоугольник {aligned.shape[1]}x{aligned.shape[0]})")
    # else:
    #     print("    (контур платы не найден — выравнивание пропущено)")

    # return best_matches, best_targets, aligned, warp_matrix


if __name__ == "__main__":
    for path in INPUT_IMAGES:
        process_image(path, OUTPUT_DIR_ROOT)