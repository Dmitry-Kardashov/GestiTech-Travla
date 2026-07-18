"""
Конвертация Gerber-файла (например, слоя дорожек F_Cu/B_Cu) в PNG.

Установка:
    pip install pygerber --break-system-packages

Использование:
    Настройте пути к файлам в блоке `if __name__ == "__main__":` и запустите скрипт:
    python gerber_to_png.py
"""

from pathlib import Path
from pygerber.gerberx3.api.v2 import GerberFile, ColorScheme, PixelFormatEnum
from pygerber.common.rgba import RGBA


def hex_to_rgba(hex_color: str, alpha: int = 255) -> RGBA:
    """Преобразует HEX-строку ('#RRGGBB') в объект RGBA."""
    hex_color = hex_color.lstrip("#")
    r, g, b = (int(hex_color[i:i + 2], 16) for i in (0, 2, 4))
    return RGBA(r=r, g=g, b=b, a=alpha)


def gerber_to_png(
    input_path: str,
    output_path: str,
    dpmm: int = 40,
    track_color: str = "#FFFFFF",  # медный цвет по умолчанию
    transparent_background: bool = True,
) -> None:
    """
    Рендерит один Gerber-слой (дорожки) в PNG.

    input_path: путь к .gbr / .gtl / .gbl и т.п. файлу одного слоя
    output_path: путь для сохранения .png
    dpmm: разрешение растра (точек на мм); больше — выше качество, крупнее файл
    track_color: цвет самих дорожек в HEX
    transparent_background: если True — фон прозрачный (RGBA PNG),
                             иначе фон будет чёрным
    """
    color_scheme = ColorScheme(
        # Основной цвет заливки (сами дорожки/паяльные площадки)
        solid_color=hex_to_rgba(track_color, alpha=255),
        solid_region_color=hex_to_rgba(track_color, alpha=255),
        # "clear" — вырезаемые (пустые) области, для одного слоя обычно совпадает
        clear_color=hex_to_rgba(track_color, alpha=255),
        clear_region_color=hex_to_rgba(track_color, alpha=255),
        # Фон: прозрачный (alpha=0) либо чёрный
        background_color=RGBA(r=0, g=0, b=0, a=0 if transparent_background else 255),
    )

    parsed = GerberFile.from_file(input_path).parse()
    parsed.render_raster(
        output_path,
        color_scheme=color_scheme,
        dpmm=dpmm,
        pixel_format=PixelFormatEnum.RGBA if transparent_background else PixelFormatEnum.RGB,
    )
    print(f"Готово: {output_path}")


if __name__ == "__main__":
    # === НАСТРОЙКИ ВНУТРИ КОДА ===
    INPUT_FILE = "input.gbr"       # Путь к твоему Gerber-файлу
    OUTPUT_FILE = "output.png"     # Куда сохранить получившийся PNG
    
    DPMM = 40                      # Разрешение (точек на мм)
    TRACK_COLOR = "#FFFFFF"        # Цвет дорожек в формате HEX (например, "#00FF00" для зелёного)
    TRANSPARENT = True             # True — прозрачный фон, False — чёрный фон
    # =============================

    gerber_to_png(
        input_path="INPUT_FILE",
        output_path="OUTPUT_FILE",
        dpmm=DPMM,
        track_color=TRACK_COLOR,
        transparent_background=TRANSPARENT,
    )