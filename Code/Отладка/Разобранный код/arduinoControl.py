import threading
import serial  
import time
import camera
# Вместо закомментированного app1, если функции склейки лежат в главном файле (например, main.py),
# то правильнее импортировать конкретную функцию или сам модуль, когда структура проекта зафиксирована.

SERIAL_PORT = '/dev/ttyUSB0' 
BAUD_RATE = 115200
last_arduino_message = "Пока нет команд от Arduino"

def arduino_listener():
    """Фоновая функция, которая постоянно слушает Serial-порт"""
    global last_arduino_message
    print("🚀 Фоновый поток прослушивания Arduino запущен...")
    
    while True:
        if arduino and arduino.is_open:
            try:
                if arduino.in_waiting > 0:
                    line = arduino.readline().decode('utf-8').strip()
                    if line:
                        print(f"📥 Получено от Arduino: {line}")
                        ParceCommand(line)
            except Exception as e:
                print(f"❌ Ошибка чтения Serial: {e}")
                time.sleep(1)
        time.sleep(0.05) 

def ParceCommand(command: str):
    """Обработчик входящих команд от микроконтроллера"""
    global last_arduino_message
    
    parts = command.split(":")
    cmd_type = parts[0]
    
    if cmd_type == "motor:complete":
        last_arduino_message = "Плата проехала, делаем снимок"
        camera.take_snapshot(camera.frame)
        
    elif cmd_type == "motor:btnstop":
        last_arduino_message = "Остановка по концевику"
        camera.take_snapshot(camera.frame)
        # ИСПРАВЛЕНО: Избегаем NameError, если app1 не импортирован
        try:
            import main # или как называется твой основной файл со stitch_all_from_folder
            main.stitch_all_from_folder()
        except ImportError:
            print("⚠️ Не удалось импортировать модуль для склейки.")
        
    else:
        last_arduino_message = f"Получена неизвестная команда: {command}"

# ИСПРАВЛЕНО: Сделали функцию универсальной, принимающей направление
def Arduino_Control(direction: str = "up", steps: int = 300):
    """Функция отправки команды движения на Arduino (добавлен \n в конец)"""
    
    # Формируем команду и ОБЯЗАТЕЛЬНО добавляем \n
    command = f"move:{direction};{steps}\n"
    
    if arduino and arduino.is_open:
        try:
            # Очищаем буферы перед отправкой
            arduino.reset_input_buffer()
            arduino.reset_output_buffer()
            
            # Отправляем команду, закодированную в байты
            arduino.write(command.encode('utf-8'))
            
            time.sleep(0.1)
            response = arduino.readline().decode('utf-8').strip()
            
            if response:
                return f"Команда '{command.strip()}' отправлена! Ответ от Arduino: {response}"
            return f"Команда '{command.strip()}' отправлена! (Arduino не вернула текстовый ответ)"
            
        except Exception as e:
            return f"Ошибка при отправке команды через Serial: {e}"
    else:
        return f"[Имитация] Порт {SERIAL_PORT} недоступен. Команда '{command.strip()}' отправлена в никуда."

# Дополнительная явная команда для движения вниз, если это необходимо для интерфейса
def Arduino_Control_Down(steps: int = 300):
    """Аналогичная команда для движения вниз"""
    return Arduino_Control(direction="down", steps=steps)

try:
    arduino = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
    time.sleep(2) 
    print(f"✅ Успешное подключение к Arduino на порту {SERIAL_PORT}")
    listener_thread = threading.Thread(target=arduino_listener, daemon=True)
    listener_thread.start()
except Exception as e:
    arduino = None
    print(f"⚠️ Не удалось подключиться к Arduino на порту {SERIAL_PORT}: {e}")
    print("Код продолжил работать в режиме имитации.")