# -*- coding: utf-8 -*-
import threading
import serial  
import time
import camera
import sys

# Принудительно переключаем stdout/stderr на UTF-8, чтобы кириллица в логах
# не превращалась в "кракозябры" при другой локали системы/консоли.
try:
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')
except AttributeError:
    pass  # старые версии Python без reconfigure - просто пропускаем

arduino = None 
last_arduino_message = "Пока нет команд от Arduino"

# Последовательность абсолютных позиций (ABS_MOVE), которые нужно пройти
# одну за другой после нажатия "Начать работу". После каждой позиции
# ждем "motor:step" от Arduino, делаем снимок и отправляем следующую позицию.
ABS_MOVE_POSITIONS = [3000, 4000, 5000, 5200]   # значения по умолчанию
current_step_index = 0        # Индекс текущей позиции в ABS_MOVE_POSITIONS

# Небольшая пауза перед снимком: мотор уже остановился, но плата по инерции
# может еще немного "дрожать"/двигаться несколько мгновений - ждем, чтобы
# кадр не получился смазанным.
SNAPSHOT_SETTLE_DELAY = 0.3  # секунды

SERIAL_PORT = '/dev/ttyUSB0'  # Убедись, что это правильный порт для твоей системы
BAUD_RATE = 115200

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
    """
    Пытается найти запущенный модуль web.py, чтобы вызвать
    web.trigger_auto_inspection() после склейки панорамы.
    Обычно web.py сам регистрирует себя в sys.modules['web'] (см. web.py).
    Здесь же — подстраховка на случай, если это не сработало: пробуем
    '__main__' (так Python называет модуль, если он запущен напрямую,
    например 'python web.py'), и проверяем, что там действительно есть
    нужная функция.
    """
    web_mod = sys.modules.get('web')
    if web_mod is not None and hasattr(web_mod, 'trigger_auto_inspection'):
        return web_mod

    main_mod = sys.modules.get('__main__')
    if main_mod is not None and hasattr(main_mod, 'trigger_auto_inspection'):
        return main_mod

    return None


def _finalize_and_stitch(reason: str):
    """
    Общая логика завершения серии снимков: сброс индекса, финальный кадр (если нужно),
    склейка панорамы и запуск автоматического анализа дефектов через web.trigger_auto_inspection.
    Вызывается и при прохождении всех позиций ABS_MOVE_POSITIONS, и при срабатывании концевика.
    """
    global current_step_index
    print(reason)
    current_step_index = 0  # Сброс индекса

    detect_mod = sys.modules.get('detect')
    web_mod = _get_web_module()

    if web_mod is None:
        print("⚠️ Не найден модуль web (нет sys.modules['web'] и в '__main__' нет trigger_auto_inspection) - "
              "автоматический анализ дефектов НЕ будет запущен, панорама будет только склеена.")

    if detect_mod and hasattr(detect_mod, 'stitch_all_from_folder'):
        print("Запуск склейки панорамы...")
        stitch_success = detect_mod.stitch_all_from_folder(web_module_ref=web_mod)
        if not stitch_success:
            print("Автоматический анализ отменен: склейка завершилась с ошибкой.")
        elif web_mod is not None:
            print("Склейка успешна, автоматический анализ дефектов запущен через web.trigger_auto_inspection().")
    else:
        print("Не удалось найти функцию stitch_all_from_folder в модуле detect.")


def ParceCommand(command: str):
    """Обработчик входящих сигналов от Arduino"""
    global last_arduino_message, current_step_index, ABS_MOVE_POSITIONS

    if command == "motor:step":
        # 1. Небольшая пауза, чтобы плата успела погасить инерционное дрожание
        #    после остановки мотора, и только потом делаем снимок свежим кадром.
        time.sleep(SNAPSHOT_SETTLE_DELAY)
        if hasattr(camera, 'current_live_frame') and camera.current_live_frame is not None:
            camera.take_snapshot(camera.current_live_frame)
            print("📸 Снимок сделан на лету!")
        else:
            print("⚠️ Кадр камеры еще не инициализирован в camera.current_live_frame.")

        # 2. Переходим к следующей позиции в списке, если она есть
        current_step_index += 1
        print(f"Итерация {current_step_index} из {len(ABS_MOVE_POSITIONS)}")

        if current_step_index < len(ABS_MOVE_POSITIONS):
            next_pos = ABS_MOVE_POSITIONS[current_step_index]
            print(f"Отправка следующей позиции ABS_MOVE:{next_pos}...")
            Arduino_Move_Abs(next_pos)
        else:
            # Все позиции пройдены - запускаем склейку и анализ.
            _finalize_and_stitch("Пройдены все заданные позиции ABS_MOVE. Запуск финальной сборки...")

    elif command == "motor:btnstop":
        # Концевик сработал раньше, чем закончились запланированные позиции
        # (аварийная/страховочная остановка) - тоже запускаем финальную сборку.
        time.sleep(SNAPSHOT_SETTLE_DELAY)
        if hasattr(camera, 'current_live_frame') and camera.current_live_frame is not None:
            camera.take_snapshot(camera.current_live_frame)

        _finalize_and_stitch("Концевик нажат. Движение остановлено. Запуск финальной сборки...")

def send_raw_command(cmd: str) -> str:
    """Вспомогательная функция отправки сырой строки в Serial"""
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
    """Отправка относительной команды движения (START_UP/START_DOWN) на микроконтроллер"""
    cmd = f"START_{direction.upper()}:{revolutions}\n"
    return send_raw_command(cmd)

def Arduino_Move_Abs(position: int):
    """Отправка команды перемещения в абсолютную позицию (ABS_MOVE:число)"""
    cmd = f"ABS_MOVE:{int(position)}\n"
    return send_raw_command(cmd)

def Stop_Motor():
    """Отправка команды экстренной/ручной остановки мотора"""
    global current_step_index
    current_step_index = 0
    return send_raw_command("move:stop\n")

def Lower_Board():
    """Отправка команды опустить плату"""
    return send_raw_command("START_DOWN:15\n")

def Motor_Calibrate():
    """Отправка команды выполнить калибровку моторов"""
    return send_raw_command("CALIBRATE\n")

def Start_Work_Routine(positions=None):
    """
    Вызывается по кнопке 'Начать работу' в Gradio.
    Запускает последовательность ABS_MOVE по списку позиций:
    отправляем positions[0], ждем motor:step (обрабатывается в ParceCommand),
    после чего оттуда же отправляется каждая следующая позиция по очереди.

    positions: список/кортеж чисел, либо строка "3000,4000,5000,5200" (из Gradio Textbox).
               Если не передано - используется ABS_MOVE_POSITIONS по умолчанию.
    """
    global ABS_MOVE_POSITIONS, current_step_index

    if positions is not None:
        if isinstance(positions, str):
            try:
                parsed = [int(p.strip()) for p in positions.split(",") if p.strip() != ""]
            except ValueError:
                return f"Ошибка: не удалось разобрать список позиций '{positions}'. Формат: 3000,4000,5000,5200"
            if not parsed:
                return "Ошибка: список позиций пуст."
            ABS_MOVE_POSITIONS = parsed
        else:
            ABS_MOVE_POSITIONS = list(positions)

    current_step_index = 0  # Сбрасываем индекс перед запуском
    print(f"Запуск рабочего цикла. Позиции ABS_MOVE: {ABS_MOVE_POSITIONS}")
    return Arduino_Move_Abs(ABS_MOVE_POSITIONS[0])

listener_thread = threading.Thread(target=arduino_listener, daemon=True)
listener_thread.start()