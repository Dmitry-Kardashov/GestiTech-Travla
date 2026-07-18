import cv2
import numpy as np
import os

# ==========================================
# ⚙️ БЛОК НАСТРОЕК (КОНФИГУРАЦИЯ)
# ==========================================
CONFIG = {
    # Пути к файлам ввода-вывода
    "path_gerber": "Code/Снимки плат для обработки/GERBER2.png",  # Укажите путь к вашему шаблону
    "path_pcb": "Code/Снимки плат для обработки/PCB_SCAN.png",  # Укажите путь к фотографии платы
    "output_dir": "Code/Отладка/Разобранный код/debugging_inspection",      # Папка для сохранения всех этапов

    # Настройки двусторонней фильтрации (Bilateral)
    "filter_d": 9,
    "sigma_color": 75,
    "sigma_space": 75,

    # Настройки бинаризации
    "block_size": 59,
    "c_val": -14,

    # Очистка маски от шума
    # Варианты: "Без очистки", "Морфологическое открытие (Быстро)", "Фильтрация по площади (Чисто)"
    "noise_method": "Морфологическое открытие (Быстро)",
    "morph_size": 4,
    "min_noise_area": 250,

    # Настройки поиска дефектов
    "min_defect_area": 170
}

# Создаем папку для сохранения промежуточных результатов
os.makedirs(CONFIG["output_dir"], exist_ok=True)


# ==========================================
# 🛠️ ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ ОБРАБОТКИ
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
    
    # 1. Используем SIFT вместо ORB для максимальной точности
    sift = cv2.SIFT_create(nfeatures=10000)

    # 2. Переводим в Grayscale
    gray_gerber = cv2.cvtColor(img_gerber, cv2.COLOR_BGR2GRAY)
    gray_pcb = cv2.cvtColor(img_pcb_aligned, cv2.COLOR_BGR2GRAY)
    
    # 3. Выравниваем контраст с помощью CLAHE
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
    gray_gerber = clahe.apply(gray_gerber)
    gray_pcb = clahe.apply(gray_pcb)

    # Находим точки и дескрипторы
    kg, dg = sift.detectAndCompute(gray_gerber, None)
    kp, dp = sift.detectAndCompute(gray_pcb, None)
    
    if dg is None or dp is None or len(kp) < 10 or len(kg) < 10:
        print("⚠️ SIFT: Недостаточно ключевых точек для анализа.")
        return img_pcb_aligned

    # 4. Для SIFT используем FLANN Matcher (работает точнее и быстрее brute-force)
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    
    matches = flann.knnMatch(dp, dg, k=2)
    
    # Фильтр Лоу (Ratio test) — отсекаем неоднозначные совпадения
    good = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good.append(m)

    if len(good) < 15:
        print(f"⚠️ SIFT: Слишком мало надежных совпадений ({len(good)}).")
        return img_pcb_aligned

    src = np.float32([kp[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst = np.float32([kg[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    
    # 5. Находим гомографию. Увеличиваем порог RANSAC до 6.0 (более мягкий отбор)
    H, inliers = cv2.findHomography(src, dst, cv2.RANSAC, 6.0)
    if H is None or inliers is None or int(inliers.sum()) < 10:
        print("⚠️ SIFT: Геометрия не сошлась даже после фильтрации RANSAC.")
        return img_pcb_aligned

    # Проверяем, чтобы трансформация не исказила картинку до неузнаваемости
    corners = np.float32([[0, 0], [gw, 0], [gw, gh], [0, gh]]).reshape(-1, 1, 2)
    moved = cv2.perspectiveTransform(corners, H).reshape(-1, 2)
    if np.linalg.norm(moved - corners.reshape(-1, 2), axis=1).max() > 0.3 * max(gw, gh):
        print("⚠️ SIFT: Матрица трансформации дает слишком сильное искажение углов. Отмена.")
        return img_pcb_aligned

    print(f"✅ Успешное точное выравнивание! Использовано точек: {int(inliers.sum())}")
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


def binarize_pcb_advanced(img_aligned, filter_d, sigma_color, sigma_space, block_size, c_val, noise_method, morph_size, min_noise_area):
    b, g, r = cv2.split(img_aligned)
    diff = cv2.subtract(r, g)
    diff = cv2.normalize(diff, None, 0, 255, cv2.NORM_MINMAX)
    filtered = cv2.bilateralFilter(
        src=diff, d=int(filter_d), sigmaColor=sigma_color, sigmaSpace=sigma_space
    )

    otsu_val, _ = cv2.threshold(filtered, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    thr = int(np.clip(otsu_val - c_val, 0, 255))
    _, binary = cv2.threshold(filtered, thr, 255, cv2.THRESH_BINARY)

    binary = cv2.morphologyEx(
        binary, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    )

    cleaned = binary.copy()
    if noise_method == "Морфологическое открытие (Быстро)":
        if morph_size > 0:
            kernel = cv2.getStructuringElement(
                cv2.MORPH_RECT, (int(morph_size), int(morph_size))
            )
            cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel)
    elif noise_method == "Фильтрация по площади (Чисто)":
        cleaned = remove_noise_by_contours(cleaned, min_noise_area)

    return cleaned


# ==========================================
# 🚀 ЕДИНАЯ ТОЧКА ЗАПУСКА И ПОСЛЕДОВАТЕЛЬНЫЙ ВЫЗОВ
# ==========================================


def smart_inspect_pcb(gerber_active, pcb_active, img_pcb_aligned, min_defect_area):
    """
    Умный анализ дефектов на основе дистанционных карт и топологического анализа.
    """
    print("[5/6] Запуск умного анализа дефектов (Distance Maps)...")
    
    # Создаем цветную копию для отрисовки результатов
    output_visual = img_pcb_aligned.copy()
    
    # -------------------------------------------------------------
    # ЧАСТЬ 1: Вычисление карт расстояний (Distance Transform)
    # -------------------------------------------------------------
    # Показывает, насколько каждая точка внутри дорожки удалена от её края
    dist_gerber = cv2.distanceTransform(gerber_active, cv2.DIST_L2, 3)
    dist_pcb = cv2.distanceTransform(pcb_active, cv2.DIST_L2, 3)

    # 1. Поиск ОБРЫВОВ и СУЖЕНИЙ (Есть на Gerber, нет на PCB)
    raw_missing = cv2.subtract(gerber_active, pcb_active)
    
    # 2. Поиск ЗАМЫКАНИЙ и ЛИШНЕЙ МЕДИ (Есть на PCB, нет на Gerber)
    raw_excess = cv2.subtract(pcb_active, gerber_active)

    # -------------------------------------------------------------
    # ЧАСТЬ 2: Умная фильтрация и классификация контуров
    # -------------------------------------------------------------
    
    # Счётчики
    stats = {"critical_breaks": 0, "warnings_narrowing": 0, "critical_shorts": 0, "minor_excess": 0}

    # --- Анализируем нехватку меди (Обрывы / Сужения) ---
    contours_missing, _ = cv2.findContours(raw_missing, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for c in contours_missing:
        area = cv2.contourArea(c)
        if area < min_defect_area:
            continue
            
        # Создаем маску конкретного дефекта
        mask_c = np.zeros_like(raw_missing)
        cv2.drawContours(mask_c, [c], -1, 255, -1)
        
        # Находим максимальную толщину дорожки в месте, где случился дефект
        # (смотрим по идеальному шаблону Gerber)
        _, max_val_gerber, _, _ = cv2.minMaxLoc(dist_gerber, mask=mask_c)
        
        x, y, w, h = cv2.boundingRect(c)
        
        # ЛОГИКА: Если дефект сожрал область, которая в Gerber была глубоко внутри проводника
        # (глубина > 4-5 пикселей от края), то это критический прогар или обрыв.
        if max_val_gerber > 4.5:
            stats["critical_breaks"] += 1
            label = f"CRIT: Break #{stats['critical_breaks']}"
            color = (0, 0, 255)  # Ярко-красный для критических
            thickness = 2
        else:
            stats["warnings_narrowing"] += 1
            label = f"WARN: Narrowing"
            color = (0, 165, 255)  # Оранжевый для сужений края
            thickness = 1

        cv2.rectangle(output_visual, (x - 3, y - 3), (x + w + 3, y + h + 3), color, thickness)
        cv2.putText(output_visual, label, (x, y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)

    # --- Анализируем избыток меди (Замыкания / Пятна) ---
    contours_excess, _ = cv2.findContours(raw_excess, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Для поиска замыканий инвертируем Gerber, чтобы понять, где "воздух"
    air_gerber = cv2.bitwise_not(gerber_active)
    dist_air = cv2.distanceTransform(air_gerber, cv2.DIST_L2, 3)

    for c in contours_excess:
        area = cv2.contourArea(c)
        if area < (min_defect_area + 10):
            continue
            
        mask_c = np.zeros_like(raw_excess)
        cv2.drawContours(mask_c, [c], -1, 255, -1)
        
        # Проверяем, насколько глубоко в зону "воздуха" залезла лишняя медь
        _, max_val_air, _, _ = cv2.minMaxLoc(dist_air, mask=mask_c)
        
        x, y, w, h = cv2.boundingRect(c)
        
        # ЛОГИКА: Если лишняя медь залезла глубоко в диэлектрик между дорожками (max_val_air высокий),
        # значит она почти или полностью перемкнула соседние трассы — это КРИТИЧЕСКОЕ замыкание.
        if max_val_air > 5.0:
            stats["critical_shorts"] += 1
            label = f"CRIT: Short #{stats['critical_shorts']}"
            color = (255, 0, 0)  # Синий / Голубой для критических замыканий
            thickness = 2
        else:
            stats["minor_excess"] += 1
            label = f"MINOR: Copper Splash"
            color = (255, 191, 0)  # Бирюзовый/Желтый для некритичной грязи
            thickness = 1

        cv2.rectangle(output_visual, (x - 3, y - 3), (x + w + 3, y + h + 3), color, thickness)
        cv2.putText(output_visual, label, (x, y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)

    # -------------------------------------------------------------
    # ЧАСТЬ 3: Сохранение аналитики
    # -------------------------------------------------------------
    # Сохраняем математические карты для отладки, нормализовав их в 0-255
    debug_dist_gerber = cv2.normalize(dist_gerber, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    cv2.imwrite(os.path.join(CONFIG["output_dir"], 'step5_distance_map.jpg'), debug_dist_gerber)

    return output_visual, stats


def main():
    print("--- СТАРТ ОБРАБОТКИ ПЛАТЫ ---")

    # Проверка наличия входных файлов
    if not os.path.exists(CONFIG["path_gerber"]) or not os.path.exists(CONFIG["path_pcb"]):
        print(f"❌ Ошибка: Убедитесь, что файлы шаблона и фото платы существуют по указанным путям:\n"
              f"Шаблон: {CONFIG['path_gerber']}\nФото: {CONFIG['path_pcb']}")
        return

    # 1. Загрузка изображений
    print("[1/6] Загрузка исходных изображений...")
    img_gerber = cv2.imread(CONFIG["path_gerber"])
    img_pcb = cv2.imread(CONFIG["path_pcb"])

    # 2. Грубое и точное выравнивание (Alignment)
    print("[2/6] Выравнивание изображения платы по шаблону...")
    img_pcb_aligned_rough = align_images_orb(img_gerber, img_pcb)
    img_pcb_aligned = refine_registration_orb(img_gerber, img_pcb_aligned_rough)
    
    # Сохраняем промежуточный этап выравнивания
    path_aligned = os.path.join(CONFIG["output_dir"], 'step2_pcb_aligned.jpg')
    cv2.imwrite(path_aligned, img_pcb_aligned)
    print(f" -> Сохранено: {path_aligned}")

    # 3. Бинаризация шаблона Gerber и платы PCB
    print("[3/6] Проведение бинаризации и сегментации меди...")
    gerber_gray = cv2.cvtColor(img_gerber, cv2.COLOR_BGR2GRAY)
    _, gerber_bin = cv2.threshold(gerber_gray, 127, 255, cv2.THRESH_BINARY)
    
    # Сохраняем промежуточный ЧБ-шаблон
    path_gerber_gray = os.path.join(CONFIG["output_dir"], 'step1_gerber_gray.jpg')
    cv2.imwrite(path_gerber_gray, gerber_bin)
    
    # Бинаризация фото платы по настройкам
    pcb_bin = binarize_pcb_advanced(
        img_pcb_aligned, 
        CONFIG["filter_d"], CONFIG["sigma_color"], CONFIG["sigma_space"], 
        CONFIG["block_size"], CONFIG["c_val"], 
        CONFIG["noise_method"], CONFIG["morph_size"], CONFIG["min_noise_area"]
    )
    
    # Сохраняем промежуточную маску меди
    path_pcb_bin = os.path.join(CONFIG["output_dir"], 'step3_pcb_binarized.jpg')
    cv2.imwrite(path_pcb_bin, pcb_bin)
    print(f" -> Сохранено: {path_pcb_bin}")

    # 4. Ограничение рабочей зоны (ROI)
    print("[4/6] Наложение маски рабочей зоны...")
    kernel_roi = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    roi_mask = cv2.dilate(gerber_bin, kernel_roi)
    
    # Сохраняем маску ROI
    path_roi = os.path.join(CONFIG["output_dir"], 'step4_roi_mask.jpg')
    cv2.imwrite(path_roi, roi_mask)

    gerber_active = cv2.bitwise_and(gerber_bin, roi_mask)
    pcb_active = cv2.bitwise_and(pcb_bin, roi_mask)

    # 5. Поиск дефектов (Обрывы и Излишки меди)
    print("[5/6] Анализ дефектов...")
    kernel_clean = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    
    # Обрывы (missing copper)
    missing_copper = cv2.bitwise_and(gerber_active, cv2.bitwise_not(pcb_active))
    missing_copper = cv2.morphologyEx(missing_copper, cv2.MORPH_OPEN, kernel_clean)
    path_missing = os.path.join(CONFIG["output_dir"], 'step5_missing_copper.jpg')
    cv2.imwrite(path_missing, missing_copper)

    # Лишняя медь (excess copper)
    excess_copper = cv2.bitwise_and(pcb_active, cv2.bitwise_not(gerber_active))
    excess_copper = cv2.morphologyEx(excess_copper, cv2.MORPH_OPEN, kernel_clean)
    path_excess = os.path.join(CONFIG["output_dir"], 'step6_excess_copper.jpg')
    cv2.imwrite(path_excess, excess_copper)

    # 6. Отрисовка дефектов на финальном изображении
    print("[6/6] Отрисовка результатов на изображении...")
    output_visual = img_pcb_aligned.copy()

    # Отрисовка обрывов (Красные рамки)
    contours_missing, _ = cv2.findContours(missing_copper, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    breaks_count = 0
    for c in contours_missing:
        if cv2.contourArea(c) > CONFIG["min_defect_area"]: 
            breaks_count += 1
            x, y, wc, hc = cv2.boundingRect(c)
            cv2.rectangle(output_visual, (x - 2, y - 2), (x + wc + 2, y + hc + 2), (0, 0, 255), 2)
            cv2.putText(output_visual, f"Break #{breaks_count}", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

    # Отрисовка лишней меди (Синие рамки)
    contours_excess, _ = cv2.findContours(excess_copper, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    extras_count = 0
    for c in contours_excess:
        if cv2.contourArea(c) > CONFIG["min_defect_area"] + 10:
            extras_count += 1
            x, y, wc, hc = cv2.boundingRect(c)
            cv2.rectangle(output_visual, (x - 2, y - 2), (x + wc + 2, y + hc + 2), (255, 0, 0), 2)
            cv2.putText(output_visual, f"Extra #{extras_count}", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)

    # Сохранение итогового результата
    result_path = os.path.join(CONFIG["output_dir"], 'PCB_INSPECTION_RESULT.jpg')
    cv2.imwrite(result_path, output_visual)

    print("\n--- ОБРАБОТКА ЗАВЕРШЕНА ---")
    print(f"Найдено обрывов: {breaks_count}")
    print(f"Найдено излишков: {extras_count}")
    print(f"Финальный результат сохранен в: '{result_path}'")
    print(f"Все промежуточные этапы находятся в папке: '{CONFIG['output_dir']}'")

# ТОЧКА ВХОДА (УБЕДИТЕСЬ, ЧТО ЗДЕСЬ НЕТ ЛИШНИХ СТРОК)
if __name__ == "__main__":
    main()

