"""
find_fiducials.py

Находит на изображении печатной платы метки-мишени вида "кольцо + точка
в центре" (как на template.png), измеряет геометрию каждой метки и САМА
автоматически делит их на две группы — "обычные" и "толстые" — по
РАЗМЕРУ ТОЧКИ ВНУТРИ КОЛЬЦА (не по толщине самого кольца — она у всех
меток практически одинаковая и для классификации не годится).
Толстые/увеличенные точки обычно используются как позиционные
(референсные) метки. Найденные метки раскрашиваются разными цветами.

Использование:
    python find_fiducials.py image.png
    python find_fiducials.py image.png --min-gap-ratio 0.1   # чувствительнее к малым различиям
    python find_fiducials.py image.png --min-radius 15 --max-radius 25

Как это работает:
1. Бинаризация изображения (Otsu, либо адаптивный порог для фото).
2. cv2.findContours с RETR_TREE — получаем полную иерархию вложенности
   контуров: внешнее кольцо (белое) -> отверстие внутри (чёрное) ->
   точка в центре (белая, "внучка" кольца в иерархии).
3. Отбираем контуры, которые круглые (circularity) и у которых есть
   дочерний контур (отверстие) с примерно тем же центром — это и есть
   метка "кольцо+точка". Если внутри отверстия есть ещё один контур
   (точка), тоже концентричный — запоминаем её радиус.
4. Для каждой метки считаем: центр, внешний радиус кольца, радиус точки
   внутри, и отношение dot_r / outer_r (это отношение не зависит от
   масштаба/разрешения фото, в отличие от абсолютного радиуса точки).
5. Отношения dot_r/outer_r всех найденных меток сортируются, ищется
   самый большой разрыв (gap) между соседними значениями. Если разрыв
   достаточно большой относительно общего разброса — метки делятся на
   две группы ("обычные" ниже разрыва, "толстая точка" выше). Если
   явного разрыва нет — все метки считаются одной группой.
   Это метод "natural breaks" для 1D-данных, порог задавать вручную
   не нужно.
6. Метки размечаются по углам платы (top-left/top-right/bottom-left/
   bottom-right) относительно общего облака найденных точек.
7. Результат рисуется поверх изображения: обычные метки — зелёным,
   метки с увеличенной точкой — красным.

Если работаете с реальным фото платы (не рендер из gerber), может
понадобиться:
- увеличить блюр (--blur) для подавления шума матрицы/пыли;
- включить --adaptive для адаптивной бинаризации при неровном освещении.
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
    inner_r: float           # radius of the black hole inside the ring
    dot_r: float | None      # radius of the white dot inside that hole (the key feature)
    dot_ratio: float | None  # dot_r / outer_r, scale-independent
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
    min_area: float = 150,
    max_area: float = 4000,
    min_circularity: float = 0.75,
    max_center_offset: float = 4.0,
    min_outer_r: float = 0.0,
    max_outer_r: float = 1e9,
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

        hole_idx = hierarchy[i][2]  # first child contour (the black hole inside the ring)
        if hole_idx == -1:
            continue
        hole = contours[hole_idx]
        if cv2.contourArea(hole) < min_area * 0.5:
            continue
        if circularity(hole) < min_circularity - 0.05:
            continue

        (ox, oy), outer_r = cv2.minEnclosingCircle(c)
        (hx, hy), inner_r = cv2.minEnclosingCircle(hole)
        if np.hypot(ox - hx, oy - hy) > max_center_offset:
            continue  # not concentric -> not our marker
        if not (min_outer_r <= outer_r <= max_outer_r):
            continue  # wrong size -> probably a component footprint, not a fiducial

        # The dot is the child of the hole (grandchild of the ring) in the hierarchy.
        dot_r = None
        dot_ratio = None
        dot_idx = hierarchy[hole_idx][2]
        if dot_idx != -1:
            dot = contours[dot_idx]
            if cv2.contourArea(dot) >= 5 and circularity(dot) >= 0.5:
                (dx, dy), dr = cv2.minEnclosingCircle(dot)
                if np.hypot(ox - dx, oy - dy) <= max_center_offset:
                    dot_r = dr
                    dot_ratio = dr / outer_r

        markers.append(Marker(
            x=ox, y=oy, outer_r=outer_r, inner_r=inner_r,
            dot_r=dot_r, dot_ratio=dot_ratio,
        ))
    return markers


def label_corners(markers: list[Marker], img_w: int, img_h: int) -> None:
    """Tag each marker with the nearest corner name based on image bounds."""
    for m in markers:
        vert = "top" if m.y < img_h / 2 else "bottom"
        horiz = "left" if m.x < img_w / 2 else "right"
        m.corner = f"{vert}-{horiz}"


def auto_classify_dot_size(markers: list[Marker], min_gap_ratio: float = 0.3) -> bool:
    """
    Automatically split markers into two groups based on the size of the dot
    inside the ring (dot_r / outer_r), with no manually chosen threshold.

    Method (max-gap / 1D natural-breaks clustering), same idea as before but
    applied to the dot-size ratio instead of ring wall thickness:
    1. Sort dot_ratio values (markers without a detected dot are skipped).
    2. Find the single biggest gap between two consecutive values.
    3. If that gap is large relative to the overall spread, split there:
       below the gap -> normal dot, above -> thick dot.
    4. If no gap is large enough, everything stays in one group.

    Returns True if a "thick dot" group was found, False otherwise.
    """
    candidates = [m for m in markers if m.dot_ratio is not None]
    for m in markers:
        m.is_thick_dot = False

    if len(candidates) < 2:
        return False

    order = sorted(candidates, key=lambda m: m.dot_ratio)
    values = [m.dot_ratio for m in order]

    value_range = values[-1] - values[0]
    if value_range <= 1e-6:
        return False

    gaps = [values[i + 1] - values[i] for i in range(len(values) - 1)]
    best_gap_idx = int(np.argmax(gaps))
    best_gap = gaps[best_gap_idx]

    if best_gap / value_range < min_gap_ratio:
        return False

    split_value = (values[best_gap_idx] + values[best_gap_idx + 1]) / 2
    for m in candidates:
        m.is_thick_dot = m.dot_ratio > split_value
    return True


def draw_debug(img_bgr: np.ndarray, markers: list[Marker], found_thick: bool) -> np.ndarray:
    marked = img_bgr.copy()
    NORMAL_COLOR = (0, 200, 0)   # green
    THICK_COLOR = (0, 0, 255)    # red
    for m in markers:
        color = THICK_COLOR if m.is_thick_dot else NORMAL_COLOR
        cv2.circle(marked, (int(m.x), int(m.y)), int(m.outer_r) + 4, color, 2)
        ratio_txt = f"{m.dot_ratio:.2f}" if m.dot_ratio is not None else "?"
        tag = "THICK-DOT" if m.is_thick_dot else "normal"
        label = f"{tag} d/r={ratio_txt}"
        text_y = int(m.y - m.outer_r - 8)
        cv2.putText(marked, label, (int(m.x) - 55, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

    # Legend goes in its own strip ABOVE the image, so it can never be
    # confused with an actual detected marker on the board itself.
    strip_h = 40
    h, w = marked.shape[:2]
    out = np.zeros((h + strip_h, w, 3), dtype=np.uint8)
    out[strip_h:, :] = marked

    cv2.putText(out, "legend:", (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.circle(out, (95, 20), 8, NORMAL_COLOR, 2)
    cv2.putText(out, "normal", (110, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.55, NORMAL_COLOR, 1, cv2.LINE_AA)
    if found_thick:
        cv2.circle(out, (210, 20), 8, THICK_COLOR, 2)
        cv2.putText(out, "thick dot", (225, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.55, THICK_COLOR, 1, cv2.LINE_AA)
    return out


def main():
    ap = argparse.ArgumentParser(description="Find ring-shaped fiducial markers on a PCB image, "
                                              "classified by the size of the dot inside them")
    ap.add_argument("image", help="Path to the PCB image")
    ap.add_argument("--adaptive", action="store_true",
                     help="Use adaptive thresholding (recommended for real photos)")
    ap.add_argument("--blur", type=int, default=0,
                     help="Gaussian blur kernel size before thresholding (e.g. 3 or 5)")
    ap.add_argument("--min-gap-ratio", type=float, default=0.3,
                     help="Sensitivity of automatic normal/thick-dot clustering: the biggest "
                          "gap between sorted dot_r/outer_r values must be at least this "
                          "fraction of the full range to count as a real second group "
                          "(default 0.3). Lower it (e.g. 0.15) to catch subtler differences, "
                          "raise it to require a more obvious one.")
    ap.add_argument("--min-area", type=float, default=150,
                     help="Minimum contour area to consider (filters out noise)")
    ap.add_argument("--max-area", type=float, default=4000,
                     help="Maximum contour area to consider (filters out large shapes)")
    ap.add_argument("--min-radius", type=float, default=0,
                     help="Minimum outer radius (px) of a valid fiducial ring")
    ap.add_argument("--max-radius", type=float, default=1e9,
                     help="Maximum outer radius (px) of a valid fiducial ring. "
                          "Use --min-radius/--max-radius to exclude similar-looking "
                          "rings that belong to component footprints, not fiducials.")
    ap.add_argument("--out", default="fiducials_annotated.png",
                     help="Path to save the annotated debug image")
    ap.add_argument("--json", default=None,
                     help="Optional path to save detected markers as JSON")
    args = ap.parse_args()

    gray = cv2.imread(args.image, cv2.IMREAD_GRAYSCALE)
    if gray is None:
        sys.exit(f"Could not read image: {args.image}")
    img_bgr = cv2.imread(args.image, cv2.IMREAD_COLOR)
    h, w = gray.shape

    bw = binarize(gray, args.blur, args.adaptive)
    markers = find_ring_markers(
        bw, min_area=args.min_area, max_area=args.max_area,
        min_outer_r=args.min_radius, max_outer_r=args.max_radius,
    )

    if not markers:
        print("Метки не найдены. Попробуйте --adaptive, --blur 3, "
              "или подстройте --min-area/--max-area/--min-radius/--max-radius "
              "под реальный размер меток на фото.")
        sys.exit(1)

    label_corners(markers, w, h)
    found_thick = auto_classify_dot_size(markers, min_gap_ratio=args.min_gap_ratio)

    markers.sort(key=lambda m: (m.y, m.x))

    print(f"Найдено меток: {len(markers)}\n")
    print(f"{'x':>8} {'y':>8} {'outer_r':>8} {'dot_r':>8} {'dot/outer':>10} {'corner':>12} {'thick dot?':>10}")
    for m in markers:
        dot_r_txt = f"{m.dot_r:.2f}" if m.dot_r is not None else "n/a"
        ratio_txt = f"{m.dot_ratio:.3f}" if m.dot_ratio is not None else "n/a"
        print(f"{m.x:8.1f} {m.y:8.1f} {m.outer_r:8.2f} {dot_r_txt:>8} "
              f"{ratio_txt:>10} {m.corner:>12} {'YES' if m.is_thick_dot else '':>10}")

    thick = [m for m in markers if m.is_thick_dot]
    if found_thick and thick:
        print(f"\nАвтоматически найдена группа меток с увеличенной точкой: {len(thick)} шт.")
        for m in thick:
            print(f"  -> ({m.x:.1f}, {m.y:.1f}) в области '{m.corner}', "
                  f"dot_r={m.dot_r:.2f}, отношение dot/outer={m.dot_ratio:.3f}")
    else:
        print("\nВторой явной группы по размеру точки не найдено — все метки "
              "похожи. Если разница всё же должна быть, попробуйте уменьшить "
              "--min-gap-ratio (например, 0.15) или проверьте --min-radius/--max-radius "
              "(возможно, часть меток отфильтровывается и не попадает в сравнение).")

    debug_img = draw_debug(img_bgr, markers, found_thick)
    cv2.imwrite(args.out, debug_img)
    print(f"\nРазмеченное изображение сохранено: {args.out}")

    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump([asdict(m) for m in markers], f, ensure_ascii=False, indent=2)
        print(f"JSON с координатами сохранён: {args.json}")


if __name__ == "__main__":
    main()