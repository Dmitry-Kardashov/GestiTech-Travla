"""
find_fiducials.py

Находит на изображении печатной платы метки-мишени вида "кольцо + точка
в центре" (референсные/технологические реперы по периметру платы),
измеряет геометрию каждой метки и САМА автоматически делит их на две
группы — "тонкие" и "толстые" — по РАЗМЕРУ ТОЧКИ ВНУТРИ КОЛЬЦА
(отношение dot_r / outer_r). Толстые (увеличенная точка) метки обычно
используются как позиционные. Тонкие метки красятся зелёным, толстые —
красным.

Использование:
    python metki.py image.png
    python metki.py image.png --min-gap-ratio 0.15   # чувствительнее к малым различиям
    python metki.py image.png --adaptive --blur 3    # для реального фото платы

Почему этот код надёжнее наивного поиска кругов:
1. Бинаризация (Otsu, либо адаптивный порог для фото).
2. cv2.findContours с RETR_TREE — берём полную иерархию вложенности:
   внешнее кольцо (белое) -> отверстие внутри (чёрное) -> точка в центре
   (белая, "внучка" кольца). Настоящая метка ОБЯЗАНА иметь все три уровня
   и общий центр — это сразу отсекает обычные пятачки, дорожки и текст.
3. Жёсткая круглость (circularity) кольца И отверстия отсекает квадратные
   BGA-сетки и элементы шелкографии, которые внешне похожи на "кольцо+точка".
4. Дедупликация: у каждого настоящего кольца его чёрное отверстие тоже
   круглое и findContours находит его как отдельное "кольцо" — этот
   дубликат (тот же центр) убирается, остаётся одна метка на мишень.
5. Пороги площади/радиуса берутся ОТ РАЗМЕРА КАРТИНКИ, а не в абсолютных
   пикселях — поэтому код работает на платах разного разрешения без правки.
6. Из кандидатов оставляем метки, которые (а) лежат в краевой зоне платы
   (реперы всегда по периметру) и (б) совпадают по размеру с основной
   массой меток. Это убирает мелкие "кольца+точки" из тест-полигона в
   центре платы.
7. Классификация тонкая/толстая — методом "natural breaks" (максимальный
   разрыв в отсортированных значениях dot_r/outer_r). Порог руками задавать
   не нужно; если явной второй группы нет — все метки считаются тонкими.

Для реального фото (не рендер из gerber) полезны --adaptive и --blur.
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
    inner_r: float           # радиус чёрного отверстия внутри кольца
    dot_r: float | None      # радиус белой точки в центре (ключевой признак)
    dot_ratio: float | None  # dot_r / outer_r, не зависит от масштаба
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
    min_circularity: float = 0.85,
    center_offset_ratio: float = 0.15,
) -> list[Marker]:
    """
    Находит все кандидаты "кольцо + точка": круглое белое кольцо, внутри
    него круглое чёрное отверстие с тем же центром, а в отверстии — белая
    точка (тоже по центру). Жёсткая круглость кольца и отверстия отсекает
    квадратные BGA-сетки и шелкографию.
    """
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

        hole_idx = hierarchy[i][2]  # первое вложенное отверстие (чёрный круг)
        if hole_idx == -1:
            continue
        hole = contours[hole_idx]
        if cv2.contourArea(hole) < min_area * 0.3:
            continue
        # Отверстие настоящей метки — круглое. У BGA-сетки/квадрата — нет.
        if circularity(hole) < min_circularity - 0.05:
            continue

        (ox, oy), outer_r = cv2.minEnclosingCircle(c)
        (hx, hy), inner_r = cv2.minEnclosingCircle(hole)
        if outer_r <= 0:
            continue
        # Соосность кольца и отверстия (в долях внешнего радиуса).
        if np.hypot(ox - hx, oy - hy) > outer_r * center_offset_ratio:
            continue

        # Точка — "внучка" кольца (потомок отверстия) и тоже по центру.
        dot_r = None
        dot_ratio = None
        dot_idx = hierarchy[hole_idx][2]
        if dot_idx != -1:
            dot = contours[dot_idx]
            if cv2.contourArea(dot) >= 3 and circularity(dot) >= 0.55:
                (dx, dy), dr = cv2.minEnclosingCircle(dot)
                if np.hypot(ox - dx, oy - dy) <= outer_r * center_offset_ratio:
                    dot_r = dr
                    dot_ratio = dr / outer_r

        # Метка обязана иметь точку в центре — иначе это просто кольцо/пятак.
        if dot_r is None:
            continue

        markers.append(Marker(
            x=ox, y=oy, outer_r=outer_r, inner_r=inner_r,
            dot_r=dot_r, dot_ratio=dot_ratio,
        ))
    return markers


def dedup_markers(markers: list[Marker]) -> list[Marker]:
    """
    У каждого настоящего кольца его чёрное отверстие findContours тоже
    отдаёт как отдельное "кольцо" (с той же серединой). Оставляем на каждую
    физическую мишень одну метку — с наибольшим внешним радиусом.
    """
    kept: list[Marker] = []
    for m in sorted(markers, key=lambda m: -m.outer_r):
        if any(np.hypot(m.x - k.x, m.y - k.y) < max(m.outer_r, k.outer_r)
               for k in kept):
            continue
        kept.append(m)
    return kept


def select_fiducials(markers: list[Marker], img_w: int, img_h: int,
                     edge_frac: float = 0.22) -> list[Marker]:
    """
    Реперы всегда по периметру платы и одного размера. Оставляем метки,
    которые (а) лежат в краевой полосе (у любого из четырёх краёв) и
    (б) совпадают по размеру с основной группой меток. Это убирает мелкие
    "кольца+точки" тест-полигона в центре платы.
    """
    if not markers:
        return []

    mx, my = img_w * edge_frac, img_h * edge_frac
    peri = [m for m in markers
            if m.x < mx or m.x > img_w - mx or m.y < my or m.y > img_h - my]
    if not peri:
        peri = markers

    # Основной размер меток — медиана внешнего радиуса краевых кандидатов.
    med = float(np.median([m.outer_r for m in peri]))
    return [m for m in peri if 0.6 * med <= m.outer_r <= 1.7 * med]


def label_corners(markers: list[Marker], img_w: int, img_h: int) -> None:
    """Проставляет каждой метке имя зоны по её положению на плате."""
    for m in markers:
        vy = m.y / img_h
        vx = m.x / img_w
        vert = "top" if vy < 0.38 else ("bottom" if vy > 0.62 else "middle")
        horiz = "left" if vx < 0.38 else ("right" if vx > 0.62 else "center")
        m.corner = f"{vert}-{horiz}"


def auto_classify_dot_size(markers: list[Marker], min_gap_ratio: float = 0.3) -> bool:
    """
    Делит метки на две группы по размеру точки (dot_r / outer_r) без ручного
    порога — методом максимального разрыва (1D natural-breaks):
    1. Сортируем значения dot_ratio.
    2. Находим самый большой разрыв между соседними.
    3. Если он велик относительно общего разброса — делим там: ниже — тонкие,
       выше — толстые. Иначе всё считается одной группой (тонкие).
    Возвращает True, если найдена группа "толстых".
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


def load_calibration(path: str):
    """Загружает матрицу камеры и коэффициенты дисторсии из .npz."""
    with np.load(path) as d:
        return d["mtx"], d["dist"]


def undistort(img_bgr: np.ndarray, mtx, dist) -> np.ndarray:
    """Убирает дисторсию объектива — иначе геометрия/координаты меток врут."""
    h, w = img_bgr.shape[:2]
    new_mtx, _ = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
    return cv2.undistort(img_bgr, mtx, dist, None, new_mtx)


def detect(img_bgr: np.ndarray, *, adaptive: bool = False, blur: int = 0,
           min_circularity: float = 0.85, edge_frac: float = 0.22,
           min_gap_ratio: float = 0.3) -> tuple[list[Marker], bool]:
    """
    Полный конвейер поиска реперов на BGR-кадре (из файла или с камеры):
    бинаризация -> поиск колец+точек -> дедуп -> отбор по периметру/размеру
    -> разметка зон -> авто-деление тонкая/толстая. Возвращает (метки, есть_толстые).
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    bw = binarize(gray, blur, adaptive)
    ia = w * h
    raw = dedup_markers(find_ring_markers(bw, ia * 0.00002, ia * 0.01, min_circularity))
    markers = select_fiducials(raw, w, h, edge_frac)
    label_corners(markers, w, h)
    found_thick = auto_classify_dot_size(markers, min_gap_ratio)
    markers.sort(key=lambda m: (m.y, m.x))
    return markers, found_thick


def draw_debug(img_bgr: np.ndarray, markers: list[Marker], found_thick: bool) -> np.ndarray:
    marked = img_bgr.copy()
    NORMAL_COLOR = (0, 200, 0)   # зелёный — тонкая точка
    THICK_COLOR = (0, 0, 255)    # красный — толстая точка
    for m in markers:
        color = THICK_COLOR if m.is_thick_dot else NORMAL_COLOR
        cv2.circle(marked, (int(m.x), int(m.y)), int(m.outer_r) + 4, color, 2)
        cv2.circle(marked, (int(m.x), int(m.y)), 2, color, -1)
        ratio_txt = f"{m.dot_ratio:.2f}" if m.dot_ratio is not None else "?"
        tag = "THICK" if m.is_thick_dot else "thin"
        label = f"{tag} d/r={ratio_txt}"
        text_y = int(m.y - m.outer_r - 8)
        cv2.putText(marked, label, (int(m.x) - 55, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

    # Легенда — в отдельной полосе НАД изображением, чтобы её нельзя было
    # спутать с настоящей меткой на плате.
    strip_h = 40
    h, w = marked.shape[:2]
    out = np.zeros((h + strip_h, w, 3), dtype=np.uint8)
    out[strip_h:, :] = marked

    cv2.putText(out, f"found: {len(markers)}", (10, 26),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.circle(out, (150, 20), 8, NORMAL_COLOR, 2)
    cv2.putText(out, "thin", (165, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.55, NORMAL_COLOR, 1, cv2.LINE_AA)
    if found_thick:
        cv2.circle(out, (240, 20), 8, THICK_COLOR, 2)
        cv2.putText(out, "thick dot", (255, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.55, THICK_COLOR, 1, cv2.LINE_AA)
    return out


def main():
    ap = argparse.ArgumentParser(description="Поиск реперных меток 'кольцо+точка' на плате, "
                                             "классификация по размеру точки")
    ap.add_argument("image", help="Путь к изображению платы")
    ap.add_argument("--adaptive", action="store_true",
                    help="Адаптивная бинаризация (рекомендуется для реальных фото)")
    ap.add_argument("--blur", type=int, default=0,
                    help="Размер ядра размытия перед бинаризацией (например, 3 или 5)")
    ap.add_argument("--min-gap-ratio", type=float, default=0.3,
                    help="Чувствительность авто-деления тонкая/толстая: самый большой "
                         "разрыв между отсортированными dot_r/outer_r должен быть не меньше "
                         "этой доли от полного разброса (по умолчанию 0.3). Меньше (0.15) — "
                         "ловит более тонкие различия.")
    ap.add_argument("--min-circularity", type=float, default=0.85,
                    help="Минимальная круглость кольца и отверстия (0..1). Выше — строже "
                         "отсекаются квадратные BGA-сетки и шелкография.")
    ap.add_argument("--edge-frac", type=float, default=0.22,
                    help="Ширина краевой полосы (доля стороны), в которой ищутся реперы. "
                         "Метки в центре платы (тест-полигон) отбрасываются.")
    ap.add_argument("--undistort", default=None,
                    help="Путь к .npz с калибровкой камеры (mtx, dist) — убрать дисторсию")
    ap.add_argument("--out", default="fiducials_annotated.png",
                    help="Куда сохранить размеченное изображение")
    ap.add_argument("--json", default=None,
                    help="Опционально: сохранить найденные метки в JSON")
    args = ap.parse_args()

    img_bgr = cv2.imread(args.image, cv2.IMREAD_COLOR)
    if img_bgr is None:
        sys.exit(f"Не удалось прочитать изображение: {args.image}")

    if args.undistort:
        mtx, dist = load_calibration(args.undistort)
        img_bgr = undistort(img_bgr, mtx, dist)

    markers, found_thick = detect(
        img_bgr, adaptive=args.adaptive, blur=args.blur,
        min_circularity=args.min_circularity, edge_frac=args.edge_frac,
        min_gap_ratio=args.min_gap_ratio,
    )

    if not markers:
        print("Метки не найдены. Попробуйте --adaptive, --blur 3, "
              "уменьшить --min-circularity (например, 0.8) "
              "или увеличить --edge-frac.")
        sys.exit(1)

    print(f"Найдено меток: {len(markers)}\n")
    print(f"{'x':>8} {'y':>8} {'outer_r':>8} {'dot_r':>8} {'dot/outer':>10} {'зона':>14} {'толстая?':>10}")
    for m in markers:
        dot_r_txt = f"{m.dot_r:.2f}" if m.dot_r is not None else "n/a"
        ratio_txt = f"{m.dot_ratio:.3f}" if m.dot_ratio is not None else "n/a"
        print(f"{m.x:8.1f} {m.y:8.1f} {m.outer_r:8.2f} {dot_r_txt:>8} "
              f"{ratio_txt:>10} {m.corner:>14} {'YES' if m.is_thick_dot else '':>10}")

    thick = [m for m in markers if m.is_thick_dot]
    if found_thick and thick:
        print(f"\nАвтоматически найдена группа меток с увеличенной точкой: {len(thick)} шт.")
        for m in thick:
            print(f"  -> ({m.x:.1f}, {m.y:.1f}) в зоне '{m.corner}', "
                  f"dot_r={m.dot_r:.2f}, dot/outer={m.dot_ratio:.3f}")
    else:
        print("\nВторой явной группы по размеру точки не найдено — все метки похожи. "
              "Если разница всё же должна быть, уменьшите --min-gap-ratio (например, 0.15).")

    debug_img = draw_debug(img_bgr, markers, found_thick)
    cv2.imwrite(args.out, debug_img)
    print(f"\nРазмеченное изображение сохранено: {args.out}")

    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump([asdict(m) for m in markers], f, ensure_ascii=False, indent=2)
        print(f"JSON с координатами сохранён: {args.json}")


if __name__ == "__main__":
    main()
