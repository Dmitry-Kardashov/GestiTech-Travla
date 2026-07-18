import numpy as np
import cv2


CALIB = "camera_calibration.npz"   # файл калибровки (None -> без коррекции дисторсии)

# --- Параметры камеры (IMX577, UVC) ---
# Поддерживаемые режимы MJPG@30fps: 1280x720, 1920x1080, 2592x1944,
# 3840x2160, 4000x3000. Больше разрешение -> больше деталей меток,
# но тяжелее обработка. 1920x1080 -- хороший баланс.
CAP_W, CAP_H = 1920, 1080
DISPLAY_WIDTH = 1280        # ширина окна на экране (кадр только МАСШТАБИРУЕТСЯ)
FOCUS_START = 100           # стартовый фокус (0..255), моторизованный объектив


def open_camera():
    cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
    if not cap.isOpened():
        raise RuntimeError("Не удалось открыть камеру.")

    # ГЛАВНОЕ: MJPG ставим ДО разрешения, иначе камера отдаёт сырой YUYV
    # и на высоком разрешении падает fps/качество.
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAP_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAP_H)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)          # без задержки на старые кадры

    # Ручной фокус: для платы на фиксированном расстоянии автофокус «дышит».
    cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
    cap.set(cv2.CAP_PROP_FOCUS, FOCUS_START)
    return cap


def main():
    cap = open_camera()
    aw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    ah = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    disp_h = int(ah * DISPLAY_WIDTH / aw)

    # Калибровка: строим карты коррекции один раз (быстрее, чем undistort в цикле).
    maps = None
    if CALIB:
        try:
            mtx, dist = load_calibration(CALIB)
            nm, _ = cv2.getOptimalNewCameraMatrix(mtx, dist, (aw, ah), 1, (aw, ah))
            maps = cv2.initUndistortRectifyMap(mtx, dist, None, nm, (aw, ah), cv2.CV_16SC2)
            print(f"Калибровка загружена: {CALIB}")
        except Exception as e:
            print(f"Калибровка не загружена ({e}) — без коррекции дисторсии.")

    print(f"Захват: {aw}x{ah} @ {cap.get(cv2.CAP_PROP_FPS):.0f} fps")
    print("q выход | a/d фокус | f автофокус | u дисторсия | p поиск меток | s снимок")

    win = "IMX577"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, DISPLAY_WIDTH, disp_h)

    cap.release()
    cv2.destroyAllWindows()

def load_calibration(path: str):
    """Загружает матрицу камеры и коэффициенты дисторсии из .npz."""
    with np.load(path) as d:
        return d["mtx"], d["dist"]

if __name__ == "__main__":
    main()
