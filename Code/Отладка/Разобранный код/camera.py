import cv2
import numpy as np
import os

CALIB = "camera_calibration1.npz"   # файл калибровки (None -> без коррекции дисторсии)

# --- Параметры камеры (IMX577, UVC) ---
# Поддерживаемые режимы MJPG@30fps: 1280x720, 1920x1080, 2592x1944,
# 3840x2160, 4000x3000. Больше разрешение -> больше деталей меток,
# но тяжелее обработка. 1920x1080 -- хороший баланс.
CAP_W, CAP_H = 1920, 1080
DISPLAY_WIDTH = 1280        # ширина окна на экране (кадр только МАСШТАБИРУЕТСЯ)
FOCUS_START = 100           # стартовый фокус (0..255), моторизованный объектив

pcb_dir = "pcb_pic"


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

def load_calibration(path: str):
    """Загружает матрицу камеры и коэффициенты дисторсии из .npz."""
    with np.load(path) as d:
        return d["mtx"], d["dist"]
    




def CameraInit():
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

    focus, autofocus, shot = FOCUS_START, False, 0
    undist, process = maps is not None, False
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Ошибка: нет кадра.")
            break

        if undist and maps is not None:          # коррекция на полном кадре
            frame = cv2.remap(frame, maps[0], maps[1], cv2.INTER_LINEAR)

        disp = cv2.resize(frame, (DISPLAY_WIDTH, disp_h), interpolation=cv2.INTER_AREA)
        
      
        cv2.imshow(win, disp)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("f"):
            autofocus = not autofocus
            cap.set(cv2.CAP_PROP_AUTOFOCUS, int(autofocus))
            if not autofocus:
                cap.set(cv2.CAP_PROP_FOCUS, focus)
        elif key in (ord("d"), ord("a")) and not autofocus:
            focus = min(255, focus + 5) if key == ord("d") else max(0, focus - 5)
            cap.set(cv2.CAP_PROP_FOCUS, focus)
        elif key == ord("u") and maps is not None:
            undist = not undist
        elif key == ord("p"):
            process = not process
        elif key == ord("s"):
            # cv2.imwrite('snapshot.png', frame)
            take_snapshot(frame)

    cap.release()
    cv2.destroyAllWindows()


def take_snapshot(frame):
    counter = 1
    while True:
            file_name = f"{counter}.jpg"
            full_path = os.path.join(pcb_dir, file_name)
            
            if not os.path.exists(full_path):
                break  # Нашли свободное имя, выходим из цикла поиска
            counter += 1

    # Сохраняем кадр
    cv2.imwrite(full_path, frame)
    print(f"Снимок сохранен: {full_path}")



if __name__ == "__main__":
    CameraInit()
