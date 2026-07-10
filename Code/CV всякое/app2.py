import cv2
import numpy as np
import os

def process_and_save_pcb(image_path, output_dir="output_results"):
    """
    Находит углы платы на изображении и сохраняет этапы обработки в файлы.
    """
    # 1. Создаем папку для результатов, если её нет
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Создана директория для сохранения: {output_dir}")

    # Загрузка изображения
    img = cv2.imread(image_path)
    if img is None:
        print(f"Ошибка: Не удалось загрузить изображение по пути {image_path}")
        return
    
    # Ресайз для стандартизации параметров фильтрации
    height, width = img.shape[:2]
    max_dimension = 1000
    if max(height, width) > max_dimension:
        scale = max_dimension / max(height, width)
        img_resized = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    else:
        img_resized = img.copy()
        scale = 1.0

    result_img = img_resized.copy()

    # 2. Предобработка
    gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)

    # 3. Детекция границ Кенни с авто-порогом Оцу
    high_thresh, _ = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    low_thresh = 0.5 * high_thresh
    edges = cv2.Canny(blurred, low_thresh, high_thresh)

    # 4. Морфологическое закрытие (соединение разрывов)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    closed_edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

    # Сохраняем промежуточные этапы для отладки геометрии/качества
    cv2.imwrite(os.path.join(output_dir, "1_canny_edges.png"), edges)
    cv2.imwrite(os.path.join(output_dir, "2_closed_edges.png"), closed_edges)

    # 5. Поиск контуров
    contours, _ = cv2.findContours(closed_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        print("Контуры не найдены. Промежуточные маски сохранены.")
        return

    # Сортировка по площади
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]
    
    best_contour = None
    best_corners = None

    for cnt in contours:
        if cv2.contourArea(cnt) < (img_resized.shape[0] * img_resized.shape[1] * 0.05):
            continue

        perimeter = cv2.arcLength(cnt, True)
        epsilon = 0.03 * perimeter
        approx = cv2.approxPolyDP(cnt, epsilon, True)

        if len(approx) == 4 and cv2.isContourConvex(approx):
            best_contour = cnt
            best_corners = approx
            break

    # Запасной вариант, если идеальный 4-угольник не найден
    if best_corners is None and len(contours) > 0:
        best_contour = contours[0]
        perimeter = cv2.arcLength(best_contour, True)
        best_corners = cv2.approxPolyDP(best_contour, 0.04 * perimeter, True)
        print("Предупреждение: Использован самый крупный контур (возможны шумы границы).")

    # 6. Отрисовка результатов и запись финальных файлов
    if best_contour is not None:
        # Рисуем контур на уменьшенной копии
        cv2.drawContours(result_img, [best_contour], -1, (0, 255, 0), 3)
        
        # Сортировка углов (TL, TR, BR, BL)
        corners_sorted = sort_corners(best_corners.reshape(-1, 2))

        labels = ["TL", "TR", "BR", "BL"]
        colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (255, 255, 0)]

        for i, (x, y) in enumerate(corners_sorted):
            # Пересчет координат обратно на оригинальный масштаб картинки
            real_x = int(x / scale)
            real_y = int(y / scale)
            
            # Точки на уменьшенной картинке (для быстрого просмотра)
            cv2.circle(result_img, (x, y), 12, colors[i], -1)
            cv2.putText(result_img, f"{labels[i]}", (x + 15, y - 15), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Точки на ОРИГИНАЛЬНОМ full-res изображении (для максимальной точности)
            # Радиус круга адаптируем под размер оригинального кадра
            circle_radius = max(int(max(height, width) * 0.01), 10)
            cv2.circle(img, (real_x, real_y), circle_radius, colors[i], -1)
            cv2.putText(img, f"{labels[i]} ({real_x},{real_y})", (real_x + 20, real_y - 20), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 3)

        # Сохраняем финальные результаты
        cv2.imwrite(os.path.join(output_dir, "3_result_resized.png"), result_img)
        cv2.imwrite(os.path.join(output_dir, "4_result_original_size.png"), img)
        
        print(f"Обработка завершена успешно!")
        print(f"Финальные файлы сохранены в папку '{output_dir}':")
        print("  - 3_result_resized.png (Плата с контуром)")
        print("  - 4_result_original_size.png (Оригинальное разрешение с точными координатами)")
    else:
        print("Не удалось выделить плату.")

def sort_corners(pts):
    """Геометрическая сортировка 4-х точек: TL, TR, BR, BL"""
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect.astype(int)

# Пример запуска:
process_and_save_pcb('PCB.png', 'my_pcb_output')