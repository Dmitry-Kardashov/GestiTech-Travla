"""
find_fiducials_fixed.py

Находит на изображении печатной платы строго 8 технологических меток-мишеней
(4 по углам и 4 по центрам сторон). Автоматически фильтрует любой мусор
(BGA-сетки, элементы шелкографии, текст) благодаря строгому контролю округлости
и геометрическому отбору. Разделяет метки на "тонкие" и "толстые" по размеру
центральной точки.
"""

import argparse
import json
import sys
from dataclasses import dataclass, asdict

import cv2
import numpy as np


@dataclass
class Marker:
    x: float
    y: float
    outer_r: float
    inner_r: float           # Радиус черного отверстия внутри кольца
    dot_r: float | None      # Радиус белой точки в самом центре
    dot_ratio: float | None  # dot_r / outer_r (пропорция)
    corner: str = ""
    is_thick_dot: bool = False


def circularity(contour) -> float:
    area = cv2.contourArea(contour)
    perim = cv2.arcLength(contour, True)
    if perim == 0:
        return 0.0
    return 4 * np.pi * area / (perim * perim)


def binarize(gray: np.ndarray, blur: int, adaptive: bool) -> np.ndarray:
    if blur > 0:
        k = blur if blur % 2 == 1 else blur + 1
        gray = cv2.GaussianBlur(gray, (k, k), 0)
    if adaptive:
        bw = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 35, -5,
        )
    else:
        _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return bw


def find_ring_markers(
    bw: np.ndarray,
    min_area: float,
    max_area: float,
    min_circularity: float = 0.88,        # Жесткий фильтр на идеальную окружность
    max_center_offset_ratio: float = 0.08, # Минимальный люфт соосности центра
) -> list[Marker]:
    contours, hierarchy = cv2.findContours(bw, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if hierarchy is None:
        return []
    hierarchy = hierarchy[0]

    markers: list[Marker] = []
    for i, c in enumerate(contours):
        area = cv2.contourArea(c)
        if not (min_area < area < max_area):
            continue
        if circularity(c) < min_circularity:
            continue

        hole_idx = hierarchy[i][2]  # Первое вложенное отверстие (черный круг)
        if hole_idx == -1:
            continue
        hole = contours[hole_idx]
        hole_area = cv2.contourArea(hole)
        
        # У правильной реперной метки площадь отверстия сбалансирована относительно кольца
        if not (area * 0.15 < hole_area < area * 0.75):
            continue
        if circularity(hole) < min_circularity - 0.03:
            continue

        (ox, oy), outer_r = cv2.minEnclosingCircle(c)
        (hx, hy), inner_r = cv2.minEnclosingCircle(hole)
        
        # Допускаемое смещение центра привязано к радиусу кольца
        if np.hypot(ox - hx, oy - hy) > (outer_r * max_center_offset_ratio):
            continue  

        # Ищем центральную точку ("внучку" внешнего кольца)
        dot_r = None
        dot_ratio = None
        dot_idx = hierarchy[hole_idx][2]
        if dot_idx != -1:
            dot = contours[dot_idx]
            if cv2.contourArea(dot) >= 3 and circularity(dot) >= 0.6:
                (dx, dy), dr = cv2.minEnclosingCircle(dot)
                if np.hypot(ox - dx, oy - dy) <= (outer_r * max_center_offset_ratio):
                    dot_r = dr
                    dot_ratio = dr / outer_r

        markers.append(Marker(
            x=ox, y=oy, outer_r=outer_r, inner_r=inner_r,
            dot_r=dot_r, dot_ratio=dot_ratio,
        ))
    return markers


def filter_strict_8_markers(markers: list[Marker], img_w: int, img_h: int) -> list[Marker]:
    """
    Оставляет строго до 8 реперных меток, выбирая только те, которые 
    находятся в целевых технологических зонах (4 угла и 4 середины сторон).
    """
    pad_w = img_w * 0.20
    pad_h = img_h * 0.20
    
    # Целевые координаты идеальных позиций по периметру
    targets = {
        "top-left": (0, 0),
        "top-center": (img_w / 2, 0),
        "top-right": (img_w, 0),
        "middle-left": (0, img_h / 2),
        "middle-right": (img_w, img_h / 2),
        "bottom-left": (0, img_h),
        "bottom-center": (img_w / 2, img_h),
        "bottom-right": (img_w, img_h)
    }
    
    best_markers = {}
    
    for name, (tx, ty) in targets.items():
        candidates = []
        for m in markers:
            # Валидация зоны: метка должна лежать в своей трети платы
            if "top" in name and m.y > pad_h: continue
            if "bottom" in name and m.y < img_h - pad_h: continue
            if "left" in name and m.x > pad_w: continue
            if "right" in name and m.x < img_w - pad_w: continue
            if "center" in name and (m.x < pad_w or m.x > img_w - pad_w): continue
            if "middle" in name and (m.y < pad_h or m.y > img_h - pad_h): continue
            
            dist = np.hypot(m.x - tx, m.y - ty)
            candidates.append((dist, m))
            
        if candidates:
            # Выбираем самый близкий к краю/центру стороны объект в этой зоне
            candidates.sort(key=lambda x: x[0])
            best_m = candidates[0][1]
            best_m.corner = name
            best_markers[name] = best_m

    return list(best_markers.values())


def classify_markers(markers: list[Marker], threshold_ratio: float = 0.35):
    """
    Классификация по фиксированному порогу геометрии.
    Тонкая точка занимает ~22% радиуса, толстая ~50%. Порог 0.35 идеален.
    """
    for m in markers:
        if m.dot_ratio is not None:
            m.is_thick_dot = m.dot_ratio > threshold_ratio
        else:
            m.is_thick_dot = False


def draw_debug(img_bgr: np.ndarray, markers: list[Marker]) -> np.ndarray:
    marked = img_bgr.copy()
    NORMAL_COLOR = (0, 250, 0)   # Зеленый для тонких меток
    THICK_COLOR = (0, 0, 255)    # Красный для толстых меток
    
    for m in markers:
        color = THICK_COLOR if m.is_thick_dot else NORMAL_COLOR
        cv2.circle(marked, (int(m.x), int(m.y)), int(m.outer_r) + 4, color, 2)
        cv2.circle(marked, (int(m.x), int(m.y)), 2, (255, 255, 0), -1)
        
        ratio_txt = f"{m.dot_ratio:.2f}" if m.dot_ratio is not None else "?"
        tag = "THICK" if m.is_thick_dot else "THIN"
        label = f"{tag} ({ratio_txt})"
        
        text_y = int(m.y - m.outer_r - 8)
        cv2.putText(marked, label, (int(m.x) - 45, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)
        cv2.putText(marked, m.corner, (int(m.x) - 45, int(m.y + m.outer_r + 14)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1, cv2.LINE_AA)

    # Верхний информационный бар
    strip_h = 40
    h, w = marked.shape[:2]
    out = np.zeros((h + strip_h, w, 3), dtype=np.uint8)
    out[strip_h:, :] = marked

    cv2.putText(out, f"Total targets: {len(markers)}/8", (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.circle(out, (180, 20), 8, NORMAL_COLOR, 2)
    cv2.putText(out, "Thin Marker", (195, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.55, NORMAL_COLOR, 1, cv2.LINE_AA)
    cv2.circle(out, (320, 20), 8, THICK_COLOR, 2)
    cv2.putText(out, "Thick Marker", (335, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.55, THICK_COLOR, 1, cv2.LINE_AA)
    return out


def main():
    ap = argparse.ArgumentParser(description="Strict 8-Fiducial Finder for PCBs")
    ap.add_argument("image", help="Path to the PCB image")
    ap.add_argument("--adaptive", action="store_true", help="Use adaptive thresholding")
    ap.add_argument("--blur", type=int, default=0, help="Blur kernel size")
    ap.add_argument("--out", default="fiducials_annotated.png", help="Path to save annotated image")
    ap.add_argument("--json", default=None, help="Path to save JSON results")
    args = ap.parse_args()

    gray = cv2.imread(args.image, cv2.IMREAD_GRAYSCALE)
    if gray is None:
        sys.exit(f"Could not read image: {args.image}")
    img_bgr = cv2.imread(args.image, cv2.IMREAD_COLOR)
    h, w = gray.shape

    # Динамический подсчет площади от разрешения картинки
    img_area = w * h
    min_area = img_area * 0.00002  
    max_area = img_area * 0.01     

    bw = binarize(gray, args.blur, args.adaptive)
    
    # Сбор всех круглых кандидатов
    raw_markers = find_ring_markers(bw, min_area=min_area, max_area=max_area)
    
    # Жесткий отбор строго 8 меток по краям платы
    markers = filter_strict_8_markers(raw_markers, w, h)

    if not markers:
        print("Метки не найдены! Проверьте файл или добавьте флаг --blur 3")
        sys.exit(1)

    classify_markers(markers, threshold_ratio=0.35)

    # Сортировка для красивого вывода в консоль
    markers.sort(key=lambda m: (m.y, m.x))

    print(f"Успешно отфильтровано и найдено технологических меток: {len(markers)} из 8\n")
    print(f"{'Позиция':<15} {'X':>8} {'Y':>8} {'Пропорция':>12} {'Тип':>10}")
    print("-" * 58)
    for m in markers:
        ratio_txt = f"{m.dot_ratio:.3f}" if m.dot_ratio is not None else "n/a"
        m_type = "THICK" if m.is_thick_dot else "THIN"
        print(f"{m.corner:<15} {m.x:8.1f} {m.y:8.1f} {ratio_txt:>12} {m_type:>10}")

    debug_img = draw_debug(img_bgr, markers)
    cv2.imwrite(args.out, debug_img)
    print(f"\nРазмеченный рендер сохранен в: {args.out}")

    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump([asdict(m) for m in markers], f, ensure_ascii=False, indent=2)
        print(f"Координаты экспортированы в JSON: {args.json}")


if __name__ == "__main__":
    main()