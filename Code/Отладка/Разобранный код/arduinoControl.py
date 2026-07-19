import threading
import serial  
import time
import camera
import sys

# ==========================================
# 🛑 ИНИЦИАЛИЗАЦИЯ ГЛОБАЛЬНЫХ ПЕРЕМЕННЫХ
# ==========================================
arduino = None 
last_arduino_message = "Пока нет команд от Arduino"

# ==========================================
# ⚙️ НАСТРОЙКА АВТОМАТИЗАЦИИ
# ==========================================
REVOLUTIONS_PER_STEP = 3.5 
SERIAL_PORT = '/dev/ttyUSB0' # Убедись, что это правильный порт для твоей системы
BAUD_RATE = 9600

def arduino_listener():
    """Фоновый поток: автоматически подключается к Arduino и слушает Serial-порт"""
    global last_arduino_message, arduino
    print("🚀 Фоновый поток управления Arduino запущен...")
    
    while True:
        # Если порт не инициализирован или закрылся (например, выдернули кабель)
        if arduino is None or not arduino.is_open:
            try:
                print(f"🔌 Попытка подключения к Arduino на порту {SERIAL_PORT}...")
                arduino = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
                time.sleep(2) # Обязательный таймаут на перезагрузку ESP32 при инициализации
                print(f"✅ Успешно подключено к Arduino на порту {SERIAL_PORT}!")
            except Exception:
                # Если платы нет, просто плавно ждем и пробуем снова в следующем цикле
                arduino = None
                time.sleep(2) 
                continue

        # Если порт успешно открыт — читаем данные
        try:
            if arduino.in_waiting > 0:
                line = arduino.readline().decode('utf-8').strip()
                if line:
                    ParceCommand(line)
        except Exception as e:
            print(f"❌ Ошибка соединения во время работы, сброс: {e}")
            try:
                arduino.close()
            except:
                pass
            arduino = None
            
        time.sleep(0.01) # Высокая отзывчивость для мгновенных снимков
def ParceCommand(command: str):
    """Обработчик входящих сигналов от Arduino"""
    global last_arduino_message
    main_module = sys.modules.get('__main__')
    
    # Двигатель прошел очередной отрезок НА ЛЕТУ
    if command == "motor:step":
        if hasattr(camera, 'current_live_frame') and camera.current_live_frame is not None:
            camera.take_snapshot(camera.current_live_frame)
            print("📸 Снимок сделан на лету!")
        else:
            print("⚠️ Кадр камеры еще не инициализирован в camera.current_live_frame.")
            
    # Приехали в самый верх/низ (Конец сканирования)
    elif command == "motor:btnstop":
        print("🛑 Концевик нажат. Движение остановлено. Запуск финальной сборки...")
        
        # Делаем финальное фото перед склейкой
        if hasattr(camera, 'current_live_frame') and camera.current_live_frame is not None:
            camera.take_snapshot(camera.current_live_frame) 
            
        if main_module and hasattr(main_module, 'stitch_all_from_folder'):
            print("⏳ Запуск склейки панорамы...")
            # 1. Запускаем склейку ОДИН раз и проверяем статус выполнения
            stitch_success = main_module.stitch_all_from_folder()
            
            # 2. Если панорама собралась успешно, передаем управление на авто-инспекцию
            if stitch_success:
                if hasattr(main_module, 'trigger_auto_inspection'):
                    print("🚀 Склейка готова. Переходим к автоматическому анализу дефектов...")
                    main_module.trigger_auto_inspection()
                else:
                    print("⚠️ Ошибка: В главном модуле отсутствует функция trigger_auto_inspection.")
            else:
                print("❌ Автоматический анализ отменен: Склейка завершилась с ошибкой.")
        else:
            print("⚠️ Не удалось найти функцию stitch_all_from_folder в главном модуле.")

def Arduino_Control(direction: str = "up", revolutions: float = 1.0):
    """Отправка команды движения на микроконтроллер с проверкой подключения"""
    global arduino
    cmd = f"START_{direction.upper()}:{revolutions}\n"
    
    if arduino is not None and arduino.is_open:
        try:
            arduino.write(cmd.encode('utf-8'))
            return f"Команда {cmd.strip()} отправлена."
        except Exception as e:
            # Выводим в лог/консоль, если отправка сорвалась
            log_msg = f"❌ Ошибка при отправке команды: {e}"
            print(log_msg)
            return log_msg
    else:
        # Выводим строгую ошибку в лог, если платы нет на момент клика
        log_msg = f"❌ Ошибка: Arduino не подключена! Проверьте USB-кабель и порт {SERIAL_PORT}."
        print(log_msg)
        return log_msg

def Start_Work_Routine():
    """Вызывается по кнопке 'Начать работу' в Gradio"""
    return Arduino_Control(direction="up", revolutions=REVOLUTIONS_PER_STEP)


# ==========================================
# 🚀 ЗАПУСК ФОНОВОГО ПОТОКА МОНИТОРИНГА
# ==========================================
# Поток стартует сразу при импорте модуля и непрерывно ждет/ищет плату
listener_thread = threading.Thread(target=arduino_listener, daemon=True)
listener_thread.start()