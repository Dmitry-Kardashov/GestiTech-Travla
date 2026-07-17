"""
Конвертация Gerber-файла (например, слоя дорожек F_Cu/B_Cu) в PNG.

Установка:
    pip install pygerber --break-system-packages

Использование:
    python gerber_to_png.py input.gbr output.png
    python gerber_to_png.py input.gbr output.png --dpmm 40 --color "#00FF00"
"""

import argparse
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Gerber -> PNG (один слой, дорожки)")
    parser.add_argument("input", help="путь к Gerber-файлу слоя")
    parser.add_argument("output", help="путь к результирующему PNG")
    parser.add_argument("--dpmm", type=int, default=40, help="разрешение, точек/мм (по умолчанию 40)")
    parser.add_argument("--color", default="#FFFFFF", help="цвет дорожек в HEX, напр. #00FF00")
    parser.add_argument(
        "--no-transparent",
        action="store_true",
        help="сделать фон чёрным вместо прозрачного",
    )
    args = parser.parse_args()

    gerber_to_png(
        input_path=args.input,
        output_path=args.output,
        dpmm=args.dpmm,
        track_color=args.color,
        transparent_background=not args.no_transparent,
    )


if __name__ == "__main__":
    main()