import cv2
import numpy as np
import os
import time

CALIB = "camera_calibration1.npz"   # файл калибровки (None -> без коррекции дисторсии)
current_live_frame = None           # Сюда OpenCV будет дублировать кадры

CAP_W, CAP_H = 1920, 1080
DISPLAY_WIDTH = 1280        
FOCUS_START = 100           

ROTATE_180 = True
pcb_dir = "pcb_pic"

def open_camera():
    cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
    if not cap.isOpened():
        raise RuntimeError("Не удалось открыть камеру.")

    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAP_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAP_H)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)          

    cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
    cap.set(cv2.CAP_PROP_FOCUS, FOCUS_START)
    return cap

def load_calibration(path: str):
    with np.load(path) as d:
        return d["mtx"], d["dist"]

def CameraInit(headless: bool = False):
    """
    :param headless: Если True, GUI-окна OpenCV выключаются (режим работы без монитора).
    """
    cap = open_camera()
    aw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    ah = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    disp_h = int(ah * DISPLAY_WIDTH / aw)

    maps = None
    if CALIB:
        try:
            mtx, dist = load_calibration(CALIB)
            nm, _ = cv2.getOptimalNewCameraMatrix(mtx, dist, (aw, ah), 1, (aw, ah))
            maps = cv2.initUndistortRectifyMap(mtx, dist, None, nm, (aw, ah), cv2.CV_16SC2)
            print(f"Калибровка загружена: {CALIB}")
        except Exception as e:
            print(f"Калибровка не загружена ({e}) — без коррекции дисторсии.")

    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Захват: {aw}x{ah} @ {fps:.0f} fps (Headless: {headless})")

    win = "IMX577"
    # Инициализируем окно только если есть монитор/GUI
    if not headless:
        print("q выход | a/d фокус | f автофокус | u дисторсия | p поиск меток | s снимок")
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(win, DISPLAY_WIDTH, disp_h)

    focus, autofocus = FOCUS_START, False
    undist, process = maps is not None, False

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Ошибка: нет кадра.")
                break

            if ROTATE_180:
                frame = cv2.rotate(frame, cv2.ROTATE_180)

            global current_live_frame
            current_live_frame = frame.copy()

            if undist and maps is not None:          
                frame = cv2.remap(frame, maps[0], maps[1], cv2.INTER_LINEAR)

            # Отрисовка и обработка клавиатуры только при наличии GUI
            if not headless:
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
                    take_snapshot(frame)
            else:
                # В headless-режиме предотвращаем 100% загрузку CPU паузкой
                time.sleep(0.01)

    except KeyboardInterrupt:
        print("\nОстановка по Ctrl+C")
    finally:
        cap.release()
        if not headless:
            cv2.destroyAllWindows()

def take_snapshot(frame):
    counter = 1
    if not os.path.exists(pcb_dir):
        os.makedirs(pcb_dir)

    while True:
        file_name = f"{counter}.jpg"
        full_path = os.path.join(pcb_dir, file_name)
        
        if not os.path.exists(full_path):
            break  
        counter += 1

    cv2.imwrite(full_path, frame)
    print(f"Снимок сохранен: {full_path}")

if __name__ == "__main__":
    # Для работы без монитора передайте headless=True
    # Или автоматически проверьте DISPLAY переменное окружение:
    is_headless = os.environ.get("DISPLAY") is None
    CameraInit(headless=is_headless)