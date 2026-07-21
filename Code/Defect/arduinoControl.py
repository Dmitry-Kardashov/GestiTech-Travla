import threading
import serial  
import time
import camera
import sys

arduino = None 
last_arduino_message = "Пока нет команд от Arduino"

REVOLUTIONS_PER_STEP = 3.5 
SERIAL_PORT = '/dev/ttyUSB0'  # Убедись, что это правильный порт для твоей системы
BAUD_RATE = 9600

def arduino_listener():
    """Фоновый поток: автоматически подключается к Arduino и слушает Serial-порт"""
    global last_arduino_message, arduino
    print("Fonoviy potok upravleniya Arduino zapushen...")  # Убрали эмодзи и сложные символы
    
    while True:
        if arduino is None or not arduino.is_open:
            try:
                print(f"Popitka podklucheniya к Arduino na portu {SERIAL_PORT}...")
                arduino = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
                time.sleep(2) 
                print(f"Uspeshno podklucheno к Arduino na portu {SERIAL_PORT}!")
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

def ParceCommand(command: str):
    """Обработчик входящих сигналов от Arduino"""
    global last_arduino_message
    
    # Ленивый импорт модулей для исключения круговой зависимости
    detect_mod = sys.modules.get('detect')
    web_mod = sys.modules.get('web')
    
    if command == "motor:step":
        if hasattr(camera, 'current_live_frame') and camera.current_live_frame is not None:
            camera.take_snapshot(camera.current_live_frame)
            print("📸 Снимок сделан на лету!")
        else:
            print("⚠️ Кадр камеры еще не инициализирован в camera.current_live_frame.")
            
    elif command == "motor:btnstop":
        print("Концевик нажат. Движение остановлено. Запуск финальной сборки...")
        
        if hasattr(camera, 'current_live_frame') and camera.current_live_frame is not None:
            camera.take_snapshot(camera.current_live_frame) 
            
        if detect_mod and hasattr(detect_mod, 'stitch_all_from_folder'):
            print("Запуск склейки панорамы...")
            # Передаем ссылку на web_mod, чтобы после склейки сработал автоанализ
            stitch_success = detect_mod.stitch_all_from_folder(web_module_ref=web_mod)
            if not stitch_success:
                print("Автоматический анализ отменен: Склейка завершилась с ошибкой.")
        else:
            print("Не удалось найти функцию stitch_all_from_folder в модуле detect.")

def Arduino_Control(direction: str = "up", revolutions: float = 1.0):
    """Отправка команды движения на микроконтроллер с проверкой подключения"""
    global arduino
    cmd = f"START_{direction.upper()}:{revolutions}\n"
    
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

def Start_Work_Routine():
    """Вызывается по кнопке 'Начать работу' в Gradio"""
    return Arduino_Control(direction="up", revolutions=REVOLUTIONS_PER_STEP)

listener_thread = threading.Thread(target=arduino_listener, daemon=True)
listener_thread.start()