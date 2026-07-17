"""
Бинаризация фото печатной платы: текстолит -> черный, медные дорожки -> белый.
Результат похож на Gerber-файл (маска дорожек).

Идея алгоритма
--------------
1. Коррекция неравномерной подсветки (деление на сильно размытую версию яркости) —
   убирает блики и тени, которые обычно ломают простые пороги по HSV.
2. Перевод в цветовое пространство Lab. Канал a* — это ось "зелёный <-> красный".
   Зелёный текстолит имеет отрицательный/низкий a*, медь (медно-оранжевый,
   золотистый, розоватый) — высокий a*. Это работает НАМНОГО стабильнее, чем
   пороги по Hue в HSV, потому что не зависит от яркости/оттенка меди
   (окисленная медь, блики, разные партии текстолита и т.д.).
3. Автоматический порог Отсу по каналу a* (+ фильтр на блики через канал V/S).
4. Морфологическая очистка: closing (заращивает разрывы в тонких дорожках),
   opening (убирает мелкий шум/соль-перец), удаление мелких "островков" по площади.
5. (Опционально) subpixel-сглаживание контуров, чтобы дорожки были ровными,
   как на Gerber.

Использование
-------------
    python pcb_binarize.py input.jpg output.png
"""

import sys
import cv2
import numpy as np


def correct_illumination(bgr, blur_ksize_frac=0.15):
    """Убирает неравномерную подсветку/блики делением на сильно размытую версию L-канала."""
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    L = lab[:, :, 0].astype(np.float32)

    h, w = L.shape
    k = int(max(h, w) * blur_ksize_frac)
    k = k + 1 if k % 2 == 0 else k
    k = max(k, 31)

    background = cv2.GaussianBlur(L, (k, k), 0)
    background = np.clip(background, 1, 255)

    # нормализуем так, чтобы средняя яркость сохранилась
    corrected = L / background * np.mean(background)
    corrected = np.clip(corrected, 0, 255).astype(np.uint8)

    lab[:, :, 0] = corrected
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def detect_glare_mask(bgr, v_thresh=235, s_thresh=40):
    """Находит пересвеченные (бликующие) пиксели: очень высокая яркость + низкая насыщенность."""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    V = hsv[:, :, 2]
    S = hsv[:, :, 1]
    glare = (V > v_thresh) & (S < s_thresh)
    return glare.astype(np.uint8) * 255


def fix_glare(bgr, v_thresh=235, s_thresh=40, inpaint_radius=3):
    """
    Убирает блики через инпейнтинг (восстановление цвета из окружения),
    вместо грубого dilate по маске меди. dilate/close на бликах — самая частая
    причина склеивания соседних дорожек/площадок с мелким шагом, поэтому здесь
    мы чиним только сам цвет, а не итоговую бинарную маску.
    """
    glare = detect_glare_mask(bgr, v_thresh, s_thresh)
    if not np.any(glare):
        return bgr
    # немного расширяем маску блика, чтобы захватить его "ореол"
    glare = cv2.dilate(glare, np.ones((3, 3), np.uint8), iterations=1)
    return cv2.inpaint(bgr, glare, inpaintRadius=inpaint_radius, flags=cv2.INPAINT_TELEA)


def binarize_pcb(bgr, min_component_area=None, close_ksize=3, open_ksize=3,
                  invert_if_needed=True, use_adaptive=False, adaptive_block_frac=0.03,
                  adaptive_C=-2):
    """
    Основная функция. Возвращает бинарную маску uint8 (255 = медь/дорожка, 0 = текстолит).
    """
    h, w = bgr.shape[:2]

    # 1. Убираем блики инпейнтингом (до любых Otsu/морфологий — это критично:
    #    старый способ "дотягивать" блик через dilate(15x15) склеивал соседние
    #    площадки с мелким шагом; инпейнтинг чинит только цвет, локально)
    bgr_fixed = fix_glare(bgr)

    # 2. Коррекция неравномерной подсветки
    bgr_corr = correct_illumination(bgr_fixed)

    # 3. Lab, канал a*
    lab = cv2.cvtColor(bgr_corr, cv2.COLOR_BGR2LAB)
    a_channel = lab[:, :, 1]

    # небольшое сглаживание перед порогом, чтобы Отсу не ловил текстуру текстолита
    a_blur = cv2.GaussianBlur(a_channel, (3, 3), 0)

    if use_adaptive:
        # Локальный (адаптивный) порог — полезен, когда яркость/оттенок меди
        # сильно меняется по площади платы. Размер окна считается от размера
        # кадра, чтобы окно было заметно больше зазора между площадками, но
        # меньше крупных областей платы.
        block = int(max(h, w) * adaptive_block_frac)
        block = block + 1 if block % 2 == 0 else block
        block = max(block, 11)
        mask = cv2.adaptiveThreshold(
            a_blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, block, adaptive_C
        )
    else:
        # Глобальный порог Отсу по a*
        _, mask = cv2.threshold(a_blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Отсу/адаптивный порог может инвертировать класс в зависимости от распределения —
    # проверяем, какой класс соответствует меди: у меди a* обычно ВЫШЕ среднего.
    mean_a_in_mask = a_channel[mask == 255].mean() if np.any(mask == 255) else 0
    mean_a_out_mask = a_channel[mask == 0].mean() if np.any(mask == 0) else 0
    if invert_if_needed and mean_a_in_mask < mean_a_out_mask:
        mask = cv2.bitwise_not(mask)

    # 4. Морфологическая очистка — специально МАЛЕНЬКИЕ ядра и по умолчанию
    #    close_ksize=3 (было 5), чтобы не склеивать соседние дорожки/пины
    #    с мелким шагом. Если на фото остаются разрывы в толстых дорожках —
    #    увеличивайте close_ksize; если склеиваются мелкие площадки —
    #    уменьшайте (вплоть до 0, т.е. отключить closing).
    if close_ksize > 0:
        k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_ksize, close_ksize))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k_close, iterations=1)
    if open_ksize > 0:
        k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_ksize, open_ksize))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k_open, iterations=1)

    # 6. Удаление мелких островков-шума по площади
    if min_component_area is None:
        min_component_area = max(20, int(0.00002 * h * w))  # авто-оценка от размера кадра

    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    clean_mask = np.zeros_like(mask)
    for i in range(1, n_labels):
        if stats[i, cv2.CC_STAT_AREA] >= min_component_area:
            clean_mask[labels == i] = 255
    mask = clean_mask

    return mask


def main():
    if len(sys.argv) < 3:
        print("Использование: python pcb_binarize.py input.jpg output.png")
        sys.exit(1)

    in_path, out_path = sys.argv[1], sys.argv[2]
    bgr = cv2.imread(in_path)
    if bgr is None:
        print(f"Не удалось открыть файл: {in_path}")
        sys.exit(1)

    mask = binarize_pcb(bgr)
    cv2.imwrite(out_path, mask)
    print(f"Готово: {out_path}")


if __name__ == "__main__":
    main()