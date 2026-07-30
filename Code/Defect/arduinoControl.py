# -*- coding: utf-8 -*-
import threading
import serial
import time
import os
import glob
import camera
import sys

# Принудительно переключаем stdout/stderr на UTF-8
try:
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')
except AttributeError:
    pass

arduino = None 
last_arduino_message = "Пока нет команд от Arduino"

# Настройки позиций по умолчанию
POS_START, POS_END, POS_COUNT = 3000, 5200, 6

def generate_positions(start, end, count):
    """Равномерно распределяет `count` позиций от start до end (включительно)."""
    count = max(2, int(count))
    step = (end - start) / (count - 1)
    return [int(round(start + step * i)) for i in range(count)]

ABS_MOVE_POSITIONS = generate_positions(POS_START, POS_END, POS_COUNT)
current_step_index = 0
SNAPSHOT_SETTLE_DELAY = 0.3

SERIAL_PORT = '/dev/ttyUSB0'
BAUD_RATE = 115200

# --- Флаги и события для Демо-режима ---
is_cycling = False
cycle_thread = None
step_completed_event = threading.Event()  # Сигнал о том, что мотор приехал в точку

def arduino_listener():
    """Фоновый поток: автоматически подключается к Arduino и слушает Serial-порт"""
    global last_arduino_message, arduino
    print("Фоновый поток управления Arduino запущен...")
    
    while True:
        if arduino is None or not arduino.is_open:
            try:
                print(f"Попытка подключения к Arduino на порту {SERIAL_PORT}...")
                arduino = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
                time.sleep(2) 
                print(f"Успешно подключено к Arduino на порту {SERIAL_PORT}!")
            except Exception:
                arduino = None
                time.sleep(2) 
                continue

        try:
            if arduino.in_waiting > 0:
                line = arduino.readline().decode('utf-8').strip()
                if line:
                    ParceCommand(line)
        except Exception as e:
            print(f"Ошибка соединения во время работы, сброс: {e}")
            try:
                arduino.close()
            except:
                pass
            arduino = None
            
        time.sleep(0.01)

def _get_web_module():
    web_mod = sys.modules.get('web')
    if web_mod is not None and hasattr(web_mod, 'trigger_auto_inspection'):
        return web_mod

    main_mod = sys.modules.get('__main__')
    if main_mod is not None and hasattr(main_mod, 'trigger_auto_inspection'):
        return main_mod

    return None

def _finalize_and_stitch(reason: str):
    global current_step_index
    print(reason)
    current_step_index = 0

    detect_mod = sys.modules.get('detect')
    web_mod = _get_web_module()

    if web_mod is None:
        print("⚠️ Не найден модуль web - автоматический анализ дефектов НЕ будет запущен.")

    if detect_mod and hasattr(detect_mod, 'stitch_all_from_folder'):
        print("Запуск склейки панорамы...")
        stitch_success = detect_mod.stitch_all_from_folder(web_module_ref=web_mod)
        if not stitch_success:
            print("Автоматический анализ отменен: склейка завершилась с ошибкой.")
        elif web_mod is not None:
            print("Склейка успешна, автоматический анализ дефектов запущен.")
    else:
        print("Не удалось найти функцию stitch_all_from_folder в модуле detect.")

def ParceCommand(command: str):
    """Обработчик входящих сигналов от Arduino"""
    global last_arduino_message, current_step_index, ABS_MOVE_POSITIONS, is_cycling

    if command == "motor:step":
        # Если включен режим демонстрации — просто отдаем сигнал потоку цикла
        if is_cycling:
            step_completed_event.set()
            return

        # --- Стандартный рабочий режим (съемка и движение) ---
        time.sleep(SNAPSHOT_SETTLE_DELAY)
        if hasattr(camera, 'current_live_frame') and camera.current_live_frame is not None:
            camera.take_snapshot(camera.current_live_frame)
            print("📸 Снимок сделан на лету!")
        else:
            print("⚠️ Кадр камеры еще не инициализирован.")

        current_step_index += 1
        print(f"Итерация {current_step_index} из {len(ABS_MOVE_POSITIONS)}")

        if current_step_index < len(ABS_MOVE_POSITIONS):
            next_pos = ABS_MOVE_POSITIONS[current_step_index]
            print(f"Отправка следующей позиции ABS_MOVE:{next_pos}...")
            Arduino_Move_Abs(next_pos)
        else:
            _finalize_and_stitch("Пройдены все заданные позиции ABS_MOVE. Запуск финальной сборки...")

    elif command == "motor:btnstop":
        if is_cycling:
            print("🛑 Концевик сработал во время демонстрации!")
            is_cycling = False
            step_completed_event.set()
            return

        time.sleep(SNAPSHOT_SETTLE_DELAY)
        if hasattr(camera, 'current_live_frame') and camera.current_live_frame is not None:
            camera.take_snapshot(camera.current_live_frame)

        _finalize_and_stitch("Концевик нажат. Движение остановлено. Запуск финальной сборки...")

def send_raw_command(cmd: str) -> str:
    global arduino
    if arduino is not None and arduino.is_open:
        try:
            arduino.write(cmd.encode('utf-8'))
            return f"Команда {cmd.strip()} отправлена."
        except Exception as e:
            log_msg = f"Ошибка при отправке команды: {e}"
            print(log_msg)
            return log_msg
    else:
        log_msg = f"Ошибка: Arduino не подключена! Проверьте USB-кабель и порт {SERIAL_PORT}."
        print(log_msg)
        return log_msg

def Arduino_Control(direction: str = "up", revolutions: float = 1.0):
    cmd = f"START_{direction.upper()}:{revolutions}\n"
    return send_raw_command(cmd)

def Arduino_Move_Abs(position: int):
    cmd = f"ABS_MOVE:{int(position)}\n"
    return send_raw_command(cmd)

def Stop_Motor():
    """Отправка команды экстренной/ручной остановки мотора"""
    global current_step_index, is_cycling
    is_cycling = False
    step_completed_event.set()  # Пробуждаем поток цикла, чтобы он завершился
    current_step_index = 0
    return send_raw_command("move:stop\n")

def Lower_Board():
    return send_raw_command("START_DOWN:15\n")

def Motor_Calibrate():
    return send_raw_command("CALIBRATE\n")

def Start_Work_Routine(positions=None):
    global ABS_MOVE_POSITIONS, current_step_index, is_cycling

    # Если был запущен демо-режим, отключаем его перед стартом реальной работы
    if is_cycling:
        is_cycling = False
        step_completed_event.set()

    if positions is not None:
        if isinstance(positions, str):
            try:
                parsed = [int(p.strip()) for p in positions.split(",") if p.strip() != ""]
            except ValueError:
                return f"Ошибка: не удалось разобрать список позиций '{positions}'."
            if not parsed:
                return "Ошибка: список позиций пуст."
            ABS_MOVE_POSITIONS = parsed
        else:
            ABS_MOVE_POSITIONS = list(positions)

    current_step_index = 0
    _clear_snapshots()

    print(f"Запуск рабочего цикла. Позиции ABS_MOVE: {ABS_MOVE_POSITIONS}")
    return Arduino_Move_Abs(ABS_MOVE_POSITIONS[0])

def _clear_snapshots():
    pcb_dir = getattr(camera, 'pcb_dir', 'pcb_pic')
    if not os.path.isdir(pcb_dir):
        return
    removed = 0
    for ext in ('*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG'):
        for path in glob.glob(os.path.join(pcb_dir, ext)):
            try:
                os.remove(path)
                removed += 1
            except OSError as e:
                print(f"Не удалось удалить старый снимок {path}: {e}")
    print(f"Папка снимков '{pcb_dir}' очищена (удалено файлов: {removed}).")

# --- Исправленный цикл демонстрации ---
def cycle_loop():
    """Фоновый цикл 0 -> 5000 -> 0 с честным ожиданием ответа от Arduino"""
    global is_cycling
    print("🔄 Запущен цикл перемещения моторов (0 -> 5000 -> 0)...")
    
    # Задаем точки циклирования (в коде ранее стояло 5000, в вызове 4900)
    TARGET_TOP = 5000
    TARGET_BOTTOM = 0

    while is_cycling:
        # --- 1. Движение вверх к 5000 ---
        print(f"Цикл: Движение к координате {TARGET_TOP}...")
        step_completed_event.clear()
        Arduino_Move_Abs(TARGET_TOP)
        
        # Ждем, пока Arduino пришлет 'motor:step' (timeout 30 сек на случай сбоя)
        if not step_completed_event.wait(timeout=30.0):
            print("⚠️ Демо-режим: Превышено время ожидания достижения верхнего положения.")
            break
            
        if not is_cycling:
            break

        time.sleep(1.0) # Небольшая пауза в верхней точке

        # --- 2. Движение вниз к 0 ---
        print(f"Цикл: Движение к координате {TARGET_BOTTOM}...")
        step_completed_event.clear()
        Arduino_Move_Abs(TARGET_BOTTOM)
        
        if not step_completed_event.wait(timeout=30.0):
            print("⚠️ Демо-режим: Превышено время ожидания достижения нижнего положения.")
            break

        time.sleep(1.0) # Небольшая пауза в нижней точке

    is_cycling = False
    print("🛑 Циклическое движение остановлено.")

def toggle_cyclic_movement():
    """Запускает или останавливает циклическое движение"""
    global is_cycling, cycle_thread
    
    if is_cycling:
        is_cycling = False
        step_completed_event.set()
        return "Циклическое движение останавливается..."
    else:
        is_cycling = True
        cycle_thread = threading.Thread(target=cycle_loop, daemon=True)
        cycle_thread.start()
        return "Запущено циклическое движение (0 - 5000)."

# Запуск слушателя
listener_thread = threading.Thread(target=arduino_listener, daemon=True)
listener_thread.start()