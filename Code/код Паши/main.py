import cv2
import numpy as np
import os

# ==========================================
# ⚙️ БЛОК НАСТРОЕК (ПЕРЕМЕННЫЕ)
# ==========================================



# Пути к файлам
INPUT_IMAGE_PATH = 'PCB.jpg'
OUTPUT_DIR = 'debugging3'
# ==========================================

# Создаем папку для сохранения промежуточных шагов
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 1. Загрузка изображения
img = cv2.imread(INPUT_IMAGE_PATH)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imwrite(os.path.join(OUTPUT_DIR, 'step1_gray.jpg'), gray)


# 2. Пороговая обработка (Бинаризация)

# вар 1
# ADAPTIVE_BLOCK_SIZE = 91   # Размер области для адаптивного порога (нечетное число)
# ADAPTIVE_C = 5            # Константа, вычитаемая из средней интенсивности
# thresh = cv2.adaptiveThreshold(
#     gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
#     cv2.THRESH_BINARY_INV, ADAPTIVE_BLOCK_SIZE, ADAPTIVE_C
# )

# вар 2
ADAPTIVE_BLOCK_SIZE = 61   # Размер области для адаптивного порога (нечетное число)
ADAPTIVE_C = 10            # Константа, вычитаемая из средней интенсивности
thresh = cv2.adaptiveThreshold(
    gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, 
    cv2.THRESH_BINARY_INV, ADAPTIVE_BLOCK_SIZE, ADAPTIVE_C
)

# # вар 3
# GLOBAL_THRESHOLD_VALUE = 65 # Порог от 0 до 255 (середина)
# _, thresh = cv2.threshold(
#     gray, GLOBAL_THRESHOLD_VALUE, 255, cv2.THRESH_BINARY_INV
# )


# Переменная порога (0) игнорируется, так как добавлен флаг cv2.THRESH_OTSU
# Использован THRESH_BINARY для прямой или THRESH_BINARY_INV для инвертированной маски
# _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)


cv2.imwrite(os.path.join(OUTPUT_DIR, 'step2_threshold.jpg'), thresh)



# 3. Поиск всех контуров
contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# Создаем копию изображения, чтобы не портить оригинал
img_all_contours = img.copy()
# Рисуем контуры: -1 означает "нарисовать ВСЕ контуры", (0, 0, 255) - красный цвет, 2 - толщина линии
cv2.drawContours(img_all_contours, contours, -1, (0, 0, 255), 1)
# Сохраняем результат в папку отладки
cv2.imwrite(os.path.join(OUTPUT_DIR, 'step3_all_raw_contours.jpg'), img_all_contours)

img = cv2.imread('PCB.jpg')
h, w, c = img.shape  # h - height, w - width, c - channels
img_all_valid_circles = img.copy()
detected_circles = []

# --- ИЗМЕНЕНИЯ ТУТ: Смягчаем фильтры круглости ---
# Рваные круги из-за рваных краев имеют огромный периметр, 
# поэтому их circularity сильно падает. Опускаем порог почти до минимума.
MIN_CIRCULARITY = 0.2     # Было 0.55. Теперь ловит даже сильно вытянутые и "зубчатые" фигуры
MAX_CIRCULARITY = 1.8     # Было 1.3. Увеличили запас на погрешности цифрового контура

# --- ИЗМЕНЕНИЯ ТУТ: Расширяем диапазоны площадей ---
# Снижаем минимальную планку, чтобы ловить совсем мелкие круги, 
# и поднимаем верхнюю, если круг "расползся".
MIN_AREA = int((w * 0.005) ** 2)    # Было 0.01 (уменьшили порог в 2 раза)
MAX_AREA = int((w * 0.15) ** 2)     # Было 0.08 (увеличили порог почти в 2 раза)

for c in contours:
    area = cv2.contourArea(c)
    
    # Дополнительный хак для рваных контуров: считаем периметр сглаженным,
    # чтобы рваные края меньше портили формулу круглости.
    # True означает, что контур замкнут. Вторым параметром (например, 1 или 2) можно сгладить контур.
    perimeter = cv2.arcLength(c, True)
    
    if perimeter == 0:
        continue
        
    # Формула круглости
    circularity = 4 * np.pi * area / (perimeter ** 2)
    
    if MIN_CIRCULARITY < circularity < MAX_CIRCULARITY and MIN_AREA < area < MAX_AREA:
        M = cv2.moments(c)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            
            # Для рваного круга расчет радиуса через площадь может быть неточным.
            # Как альтернатива: можно брать среднее между шириной и высотой ограничивающего прямоугольника:
            # _, _, width, height = cv2.boundingRect(c)
            # radius = int((width + height) / 4)
            radius = int(np.sqrt(area / np.pi))
            
            detected_circles.append((cX, cY, radius))
            cv2.circle(img_all_valid_circles, (cX, cY), radius, (255, 0, 0), 2)

cv2.imwrite(os.path.join(OUTPUT_DIR, 'step4_detected_circles.jpg'), img_all_valid_circles)


###
# ==========================================
# ⚙️ НАСТРОЙКА ДОПУСКА ДЛЯ ЗОН (В ПИКСЕЛЯХ)
# ==========================================
# Насколько далеко маркер может сместиться от своего идеального края/угла
# Если разрешение камеры очень большое, увеличьте это значение (например, до 150-200)
ZONE_TOLERANCE = 40 
# ==========================================

h, w = img.shape[:2]
4
# 1. Определяем координаты 8 идеальных целевых позиций на плате
# (В зависимости от геометрии платы, боковые метки обычно находятся строго посередине сторон)
target_positions = {
    "Верхний-Левый":   (0, 0),
    "Верхний-Центр":   (w // 2, 0),
    "Верхний-Правый":  (w, 0),
    "Левый-Центр":     (0, h // 2),
    "Правый-Центр":    (w, h // 2),
    "Нижний-Левый":    (0, h),
    "Нижний-Центр":    (w // 2, h),
    "Нижний-Правый":   (w, h)
}

img_final = img.copy()
detected_markers_by_zones = {}

# 2. Распределяем найденные круги по 8 зонам
for zone_name, target_pos in target_positions.items():
    valid_candidatesInZone = []
    
    for circle in detected_circles:
        cX, cY, radius = circle
        
        # Считаем евклидово расстояние от найденного круга до центра целевой зоны
        # Но проверяем отдельно по осям X и Y, чтобы логика "прижатия к краю" работала точнее
        dist_x = abs(cX - target_pos[0])
        dist_y = abs(cY - target_pos[1])
        
        # Модифицируем проверку: для углов важны обе координаты, для боковых — только прижатие к своему краю
        if zone_name in ["Верхний-Левый", "Верхний-Правый", "Нижний-Левый", "Нижний-Правый"]:
            is_inside_zone = (dist_x < ZONE_TOLERANCE * 2) and (dist_y < ZONE_TOLERANCE * 2)
        elif zone_name in ["Верхний-Центр", "Нижний-Центр"]:
            is_inside_zone = (dist_y < ZONE_TOLERANCE) and (abs(cX - w//2) < w//4)
        else: # Левый-Центр и Правый-Центр
            is_inside_zone = (dist_x < ZONE_TOLERANCE) and (abs(cY - h//2) < h//4)
            
        if is_inside_zone:
            # Считаем итоговое расстояние для поиска наилучшего кандидата внутри зоны
            total_dist = np.sqrt((cX - target_pos[0])**2 + (cY - target_pos[1])**2)
            valid_candidatesInZone.append((total_dist, circle))
            
    # Если в этой зоне нашли подходящие круги, выбираем самый близкий к идеальной точке
    if valid_candidatesInZone:
        valid_candidatesInZone.sort(key=lambda x: x[0]) # Сортируем по расстоянию
        best_circle = valid_candidatesInZone[0][1]     # Берем самый близкий круг
        detected_markers_by_zones[zone_name] = best_circle

# 3. Отрисовка и вывод информации
print("\n--- Результаты распознавания меток ---")
for zone_name, target_pos in target_positions.items():
    if zone_name in detected_markers_by_zones:
        cX, cY, radius = detected_markers_by_zones[zone_name]
        print(f"[ НАЙДЕН ] {zone_name:15} -> Координаты: ({cX}, {cY}), Радиус: {radius}")
        
        # Отрисовка найденных меток (Зеленый цвет)
        cv2.circle(img_final, (cX, cY), radius + 5, (0, 255, 0), 3)
        cv2.circle(img_final, (cX, cY), 3, (0, 0, 255), -1)
        # Пишем текст над маркером
        cv2.putText(img_final, zone_name, (cX - 40, cY - radius - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    else:
        print(f"[ОТСУТСТВУЕТ] {zone_name:15}")
        # Рисуем крестик там, где метка ДОЛЖНА БЫЛА быть (Красный цвет)
        tx, ty = target_pos
        # Сдвигаем маркеры от самых краев кадра вглубь на 20 пикселей для видимости отладки
        dx = 20 if tx == 0 else (-20 if tx == w else 0)
        dy = 20 if ty == 0 else (-20 if ty == h else 0)
        cv2.drawMarker(img_final, (tx + dx, ty + dy), (0, 0, 255), cv2.MARKER_CROSS, 20, 2)

# СОХРАНЯЕМ ШАГ 5: Финальный результат со всеми зонами
cv2.imwrite(os.path.join(OUTPUT_DIR, 'step5_final_corners.jpg'), img_final)

###


# =====================================================================
# 📐 БЛОК ВЫРАВНИВАНИЯ ПЕРСПЕКТИВЫ (ДОБАВИТЬ В КОНЕЦ СКРИПТА)
# =====================================================================

# 1. Задаем размеры "идеального" выровненного изображения, которое хотим получить.
# Можно взять размеры исходного фото или задать фиксированные (например, под пропорции платы)
dst_w = w
dst_h = h