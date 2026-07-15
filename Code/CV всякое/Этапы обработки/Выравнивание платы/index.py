import cv2
import numpy as np

def get_four_corners(contour):
    """
    Надежно находит 4 крайние точки контура (углы),
    игнорируя мелкие выступы и дефекты реза[cite: 52, 56].
    """
    hull = cv2.convexHull(contour)
    pts = hull.reshape(-1, 2)
    
    rect = np.zeros((4, 2), dtype="float32")
    
    # Левый-верхний угол имеет минимальную сумму (x + y)
    # Правый-нижний угол имеет максимальную сумму (x + y)
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    
    # Правый-верхний угол имеет минимальную разность (y - x)
    # Левый-нижний угол имеет максимальную разность (y - x)
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    
    return rect

def is_already_aligned(rect, image_shape, area_ratio):
    """
    Проверяет, является ли изображение уже выровненной платой.
    """
    # 1. Если плата занимает почти весь кадр (>88%), она уже обрезана
    if area_ratio > 0.88:
        return True
        
    # 2. Если плата занимает значительную часть (>60%), проверим её перекос
    h, w = image_shape[:2]
    tl, tr, br, bl = rect
    
    # Вычисляем относительный наклон верхних/нижних граней по оси Y
    top_tilt = abs(tl[1] - tr[1]) / h
    bottom_tilt = abs(bl[1] - br[1]) / h
    
    # Вычисляем относительный наклон боковых граней по оси X
    left_tilt = abs(tl[0] - bl[0]) / w
    right_tilt = abs(tr[0] - br[0]) / w
    
    # Находим максимальный перекос среди всех 4 сторон
    max_tilt = max(top_tilt, bottom_tilt, left_tilt, right_tilt)
    
    # Если перекос меньше 2.5% (линии параллельны краям фото), плата уже ровная!
    if area_ratio > 0.60 and max_tilt < 0.025:
        return True
        
    return False

def align_pcb_smart(image_path):
    # 1. Загрузка изображения
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Файл {image_path} не найден!")
    
    orig = image.copy()
    total_area = image.shape[0] * image.shape[1]
    
    # 2. Переход в HSV и выделение канала Насыщенности [cite: 49]
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    saturation = hsv[:, :, 1]
    
    # 3. Размытие и бинаризация по Оцу
    blurred = cv2.GaussianBlur(saturation, (9, 9), 0)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # 4. Морфологическое закрытие (ядро 25x25 для склеивания дорожек в сплошной силуэт) [cite: 51]
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 25))
    mask = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (11, 11)))
    
    # Сохраняем маску для отладки [cite: 54]
    cv2.imwrite("debug_mask_hsv.jpg", mask)
    
    # 5. Поиск контуров
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        print("⚠️ Контуры не найдены! Возвращаем исходное изображение.")
        return orig
        
    # Берем самый крупный контур (плата)
    pcb_contour = max(contours, key=cv2.contourArea)
    contour_area = cv2.contourArea(pcb_contour)
    area_ratio = contour_area / total_area
    
    # 6. Находим 4 угла 
    rect = get_four_corners(pcb_contour)
    
    # 7. УМНЫЙ ПРЕДОХРАНИТЕЛЬ: Проверяем, нужно ли вообще выравнивать плату?
    if is_already_aligned(rect, image.shape, area_ratio):
        print(f"✅ Плата уже выровнена (площадь: {area_ratio*100:.1f}% кадра, перекос отсутствует). Пропускаем обрезку!")
        return orig
    
    print(f"🔧 Обнаружен наклон платы (площадь: {area_ratio*100:.1f}%). Выполняем перспективное преобразование...")
    
    # 8. Вычисляем габариты и выполняем перспективное преобразование 
    (tl, tr, br, bl) = rect
    width_a = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    width_b = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    max_width = max(int(width_a), int(width_b))
    
    height_a = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    height_b = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    max_height = max(int(height_a), int(height_b))
    
    dst = np.array([
        [0, 0],
        [max_width - 1, 0],
        [max_width - 1, max_height - 1],
        [0, max_height - 1]
    ], dtype="float32")
    
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(orig, M, (max_width, max_height))
    
    return warped

if __name__ == "__main__":
    # Вы можете передать сюда как фото на столе (PCB6.jpg), так и готовый скан — код сам решит, что делать!
    result = align_pcb_smart("PCB.jpg")
    
    if result is not None:
        cv2.imwrite("aligned_pcb.jpg", result)
        print("Готово! Результат сохранен в 'aligned_pcb.jpg'.")