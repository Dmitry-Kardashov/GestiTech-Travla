import cv2
import numpy as np

def get_homography(img_src, img_dst):
    """Находит гомографию между двумя соседними изображениями"""
    sift = cv2.SIFT_create()
    kp_src, des_src = sift.detectAndCompute(img_src, None)
    kp_dst, des_dst = sift.detectAndCompute(img_dst, None)
    
    bf = cv2.BFMatcher()
    # Используем KNN-матчинг для фильтрации по методу Лоу
    matches = bf.knnMatch(des_src, des_dst, k=2)
    
    good = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good.append(m)
            
    if len(good) < 4:
        raise ValueError("Недостаточно общих точек между соседними кадрами для вычисления гомографии!")
        
    src_pts = np.float32([kp_src[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp_dst[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    
    H, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    return H

def stitch_vertical_photos(image_paths, output_path='dynamic_panorama.jpg'):
    num_images = len(image_paths)
    if num_images < 2:
        print("Для сшивки нужно как минимум 2 изображения!")
        return

    print(f"Загрузка {num_images} изображений...")
    # Загружаем изображения (предполагается порядок снизу вверх)
    images = [cv2.imread(path) for path in image_paths]
    
    # 1. Выбираем индекс базового (опорного) кадра ровно посередине списка
    base_idx = num_images // 2
    print(f"Опорным кадром (базой) выбран кадр №{base_idx + 1} (индекс {base_idx})")

    # 2. Считаем гомографии между ВСЕМИ соседними парами
    # H_neighbors[i] будет хранить гомографию от кадра i к кадру i+1
    H_neighbors = {}
    for i in range(num_images - 1):
        print(f"Поиск связей между кадром {i+1} и {i+2}...")
        H_neighbors[i] = get_homography(images[i], images[i+1])

    # 3. Приводим все гомографии к единой базе (base_idx) через цепное перемножение
    homographies = [None] * num_images
    homographies[base_idx] = np.eye(3)  # База проецируется сама в себя

    # Двигаемся вниз от базы к началу (img_idx -> base_idx)
    for i in range(base_idx - 1, -1, -1):
        # Чтобы перейти от i к base_idx, нужно умножить гомографию (i+1 -> base_idx) на (i -> i+1)
        homographies[i] = np.dot(homographies[i+1], H_neighbors[i])

    # Двигаемся вверх от базы к концу (img_idx -> base_idx)
    for i in range(base_idx + 1, num_images):
        # Нам нужна обратная гомография для движения назад к базе: inv(H_neighbors[i-1])
        H_inv = np.linalg.inv(H_neighbors[i-1])
        homographies[i] = np.dot(homographies[i-1], H_inv)

    # 4. Рассчитываем размеры итогового холста (canvas)
    all_corners = []
    for i, img in enumerate(images):
        h, w = img.shape[:2]
        corners = np.float32([[0, 0], [0, h], [w, h], [w, 0]]).reshape(-1, 1, 2)
        transformed_corners = cv2.perspectiveTransform(corners, homographies[i])
        all_corners.append(transformed_corners)

    all_corners = np.concatenate(all_corners, axis=0)
    [x_min, y_min] = np.int32(all_corners.min(axis=0).ravel() - 0.5)
    [x_max, y_max] = np.int32(all_corners.max(axis=0).ravel() + 0.5)

    # 5. Матрица сдвига, чтобы не было отрицательных координат на холсте
    translation_dist = [-x_min, -y_min]
    H_translation = np.array([[1, 0, translation_dist[0]], 
                              [0, 1, translation_dist[1]], 
                              [0, 0, 1]])

    canvas_width = x_max - x_min
    canvas_height = y_max - y_min
    print(f"Размер итогового холста: {canvas_width}x{canvas_height} px")

    # 6. Трансформируем все кадры с учетом сдвига холста
    warped_images = []
    for i, img in enumerate(images):
        H_translated = np.dot(H_translation, homographies[i])
        warped = cv2.warpPerspective(img, H_translated, (canvas_width, canvas_height))
        warped_images.append(warped)

    # 7. Склеиваем кадры на одном холсте
    # Порядок наложения важен: сначала накладываем самые дальние от центра кадры, 
    # а сам базовый кадр (и его ближайших соседей) накладываем последними сверху, 
    # чтобы центральная, самая качественная часть панорамы была на переднем плане.
    result = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)
    
    # Сортируем индексы по удаленности от базового индекса (сначала самые дальние)
    render_order = sorted(range(num_images), key=lambda x: abs(x - base_idx), reverse=True)

    for idx in render_order:
        img_warped = warped_images[idx]
        mask = (img_warped > 0)
        result[mask] = img_warped[mask]

    # Сохраняем результат
    cv2.imwrite(output_path, result)
    print(f"Готово! Панорама сохранена в: {output_path}")

# --- ПРИМЕР ИСПОЛЬЗОВАНИЯ ---
if __name__ == "__main__":
    # Сюда ты можешь передать абсолютно любой список файлов (хоть 3, хоть 5, хоть 8 штук)
    # Главное — указывать их строго по порядку снизу вверх!
    my_photos = [
        'img1.jpg',  # Низ
        'img2.jpg',
        'img3.jpg'
        # 'img4.jpg',
        # 'img5.jpg',
        # 'img6.jpg'   # Верх
    ]
    
    stitch_vertical_photos(my_photos, output_path='my_vertical_panorama.jpg')