import cv2
import numpy as np
import os

# ==========================================
# ⚙️ БЛОК НАСТРОЕК
# ==========================================
INPUT_GERBER = 'GERBER.png'
INPUT_PCB = 'PCB.jpg'
# INPUT_PCB = 'PCB7.png'
OUTPUT_DIR_ROOT = 'debugging_inspection'

def align_images_orb(img_gerber, img_pcb):
    """
    Автоматическое выравнивание платы по герберу с помощью ключевых точек ORB.
    Устойчиво к поворотам на 90/180 градусов, масштабированию и растяжениям.
    """
    # Переводим в оттенки серого
    gray_gerber = cv2.cvtColor(img_gerber, cv2.COLOR_BGR2GRAY)
    gray_pcb = cv2.cvtColor(img_pcb, cv2.COLOR_BGR2GRAY)

    # Инициализируем детектор ORB (ищем до 5000 точек для высокой точности)
    orb = cv2.ORB_create(nfeatures=5000)
    
    kp_gerber, des_gerber = orb.detectAndCompute(gray_gerber, None)
    kp_pcb, des_pcb = orb.detectAndCompute(gray_pcb, None)

    if des_gerber is None or des_pcb is None:
        raise ValueError("Не удалось извлечь дескрипторы точек. Проверь входные изображения.")

    # Матчинг точек через Hamming distance
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des_gerber, des_pcb)
    
    # Сортируем по расстоянию (лучшие совпадения в начале)
    matches = sorted(matches, key=lambda x: x.distance)

    # Берем топ-100 лучших совпадений
    good_matches = matches[:100]

    if len(good_matches) < 10:
        raise ValueError(f"Слишком мало общих точек соприкосновения ({len(good_matches)}). Выравнивание невозможно.")

    # Собираем координаты совпавших точек
    pts_gerber = np.float32([kp_gerber[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    pts_pcb = np.float32([kp_pcb[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    # Находим матрицу гомографии с фильтрацией выбросов (RANSAC)
    H, mask = cv2.findHomography(pts_pcb, pts_gerber, cv2.RANSAC, 5.0)
    
    # Трансформируем фото платы к геометрии гербера
    gh, gw = img_gerber.shape[:2]
    img_pcb_aligned = cv2.warpPerspective(img_pcb, H, (gw, gh))
    
    return img_pcb_aligned

def binarize_pcb(img_aligned):
    """ Адаптивное выделение меди на основе локального контраста """
    b, g, r = cv2.split(img_aligned)
    
    # Медь сильнее отражает красный канал, подложка — зеленый
    diff = cv2.subtract(r, g)
    diff = cv2.normalize(diff, None, 0, 255, cv2.NORM_MINMAX)
    
    # Бинаризация Гаусса
    thresh = cv2.adaptiveThreshold(
        diff, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY, 51, -15
    )
    
    # Убираем шумы
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    return thresh

def inspect_pcb(gerber_path, pcb_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    img_gerber = cv2.imread(gerber_path)
    img_pcb = cv2.imread(pcb_path)
    
    if img_gerber is None or img_pcb is None:
        print("[ОШИБКА] Проверь пути к файлам GERBER.png и PCB.jpg")
        return

    print("[1/3] Поиск фич и геометрическое выравнивание слоев...")
    try:
        img_pcb_aligned = align_images_orb(img_gerber, img_pcb)
        cv2.imwrite(os.path.join(output_dir, 'step1_pcb_aligned.jpg'), img_pcb_aligned)
    except Exception as e:
        print(f"[КРИТИЧЕСКАЯ ОШИБКА ВЫРАВНИВАНИЯ]: {e}")
        return

    print("[2/3] Бинаризация и сегментация меди...")
    gerber_gray = cv2.cvtColor(img_gerber, cv2.COLOR_BGR2GRAY)
    _, gerber_bin = cv2.threshold(gerber_gray, 127, 255, cv2.THRESH_BINARY)
    
    pcb_bin = binarize_pcb(img_pcb_aligned)
    cv2.imwrite(os.path.join(output_dir, 'step2_pcb_binarized.jpg'), pcb_bin)

    # Буферная маска 15x15 (сглаживает мелкие сдвиги на краях дорожек)
    kernel_roi = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    roi_mask = cv2.dilate(gerber_bin, kernel_roi)

    gerber_active = cv2.bitwise_and(gerber_bin, roi_mask)
    pcb_active = cv2.bitwise_and(pcb_bin, roi_mask)

    print("[3/3] Дифференциальный анализ (Обрывы и Лишние фрагменты)...")
    kernel_clean = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    
    # ОБРЫВЫ (В шаблоне есть, на плате нет)
    missing_copper = cv2.bitwise_and(gerber_active, cv2.bitwise_not(pcb_active))
    missing_copper = cv2.morphologyEx(missing_copper, cv2.MORPH_OPEN, kernel_clean)

    # ЛИШНИЕ ФРАГМЕНТЫ (На плате есть, в шаблоне нет)
    excess_copper = cv2.bitwise_and(pcb_active, cv2.bitwise_not(gerber_active))
    excess_copper = cv2.morphologyEx(excess_copper, cv2.MORPH_OPEN, kernel_clean)

    output_visual = img_pcb_aligned.copy()
    MIN_DEFECT_AREA = 50  # Фильтр пиксельной грязи (игнорируем все что меньше 50 пикселей)

    # Отрисовка ОБРЫВОВ (Красные рамки)
    contours_missing, _ = cv2.findContours(missing_copper, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in contours_missing:
        if cv2.contourArea(c) > MIN_DEFECT_AREA:  
            x, y, wc, hc = cv2.boundingRect(c)
            cv2.rectangle(output_visual, (x - 2, y - 2), (x + wc + 2, y + hc + 2), (0, 0, 255), 2)
            cv2.putText(output_visual, "Break", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

    # Отрисовка ЛИШНИХ ФРАГМЕНТОВ (Синие рамки)
    contours_excess, _ = cv2.findContours(excess_copper, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in contours_excess:
        if cv2.contourArea(c) > MIN_DEFECT_AREA:
            x, y, wc, hc = cv2.boundingRect(c)
            cv2.rectangle(output_visual, (x - 2, y - 2), (x + wc + 2, y + hc + 2), (255, 0, 0), 2)
            cv2.putText(output_visual, "Extra", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)

    cv2.imwrite(os.path.join(output_dir, 'PCB_INSPECTION_RESULT.jpg'), output_visual)
    print(f"\n[ГОТОВО] Результат инспекции сохранен в: '{output_dir}/PCB_INSPECTION_RESULT.jpg'")

if __name__ == "__main__":
    inspect_pcb(INPUT_GERBER, INPUT_PCB, OUTPUT_DIR_ROOT)