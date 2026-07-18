import threading
import serial  # Добавляем библиотеку для работы с Serial
import time
import camera

# Инициализация Serial-порта. 
# Замените 'COM3' (для Windows) или '/dev/ttyUSB0' (для Linux/macOS) на порт вашей Arduino
SERIAL_PORT = '/dev/ttyUSB0' 
BAUD_RATE = 9600
last_arduino_message = "Пока нет команд от Arduino"



def arduino_listener():
    """Фоновая функция, которая постоянно слушает Serial-порт"""
    global last_arduino_message
    print("🚀 Фоновый поток прослушивания Arduino запущен...")
    
    while True:
        if arduino and arduino.is_open:
            try:
                if arduino.in_waiting > 0:
                    # Читаем строку, декодируем и убираем пробелы/переносы
                    line = arduino.readline().decode('utf-8').strip()
                    if line:
                        print(f"📥 Получено от Arduino: {line}")
                        
                        # Роутер/Обработчик команд
                        ParceCommand(line)
                        
            except Exception as e:
                print(f"❌ Ошибка чтения Serial: {e}")
                time.sleep(1)
        time.sleep(0.05) # Небольшая пауза, чтобы не перегружать процессор

def ParceCommand(command: str):
    """Собственно обработчик входящих команд от микроконтроллера"""
    global last_arduino_message
    
    # Разделяем команду и аргументы, если они есть (например, "TEMP:24")
    parts = command.split(":")
    cmd_type = parts[0]
    
    if cmd_type == "motor:complete":
        last_arduino_message = "Плата проехала, делаем снимок"
        # Здесь можно автоматически вызвать какую-то логику Python
        camera.take_snapshot(camera.frame)
        
    elif cmd_type == "motor:btnstop":
        last_arduino_message = "Остановка по концевику"
        
    else:
        last_arduino_message = f"Получена неизвестная команда: {command}"


def Arduino_Control():
    """Функция для кнопки 'Начать работу', отправляющая команду на Arduino"""
    command = "START\n" # Строка команды, которую ждет Arduino (символ \n важен для завершения строки)
    command = "motor1:up;" + str(300)
    
    if arduino and arduino.is_open:
        try:
            # Очищаем буферы перед отправкой
            arduino.reset_input_buffer()
            arduino.reset_output_buffer()
            
            # Отправляем команду, закодированную в байты
            arduino.write(command.encode('utf-8'))
            
            # Опционально: ждем ответ от Arduino (например, если она должна прислать "OK")
            time.sleep(0.1)
            response = arduino.readline().decode('utf-8').strip()
            
            if response:
                return f"Команда '{command.strip()}' отправлена! Ответ от Arduino: {response}"
            return f"Команда '{command.strip()}' отправлена! (Arduino не вернула текстовый ответ)"
            
        except Exception as e:
            return f"Ошибка при отправке команды через Serial: {e}"
    else:
        return f"[Имитация] Порт {SERIAL_PORT} недоступен. Команда '{command.strip()}' отправлена в никуда."
    


try:
    # timeout=1 нужен, чтобы скрипт не зависал при чтении, если Arduino не ответит
    arduino = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
    time.sleep(2) # Обязательная пауза, так как при открытии порта Arduino перезагружается
    print(f"✅ Успешное подключение к Arduino на порту {SERIAL_PORT}")
    listener_thread = threading.Thread(target=arduino_listener, daemon=True)
    listener_thread.start()
except Exception as e:
    arduino = None
    print(f"⚠️ Не удалось подключиться к Arduino на порту {SERIAL_PORT}: {e}")
    print("Код продолжит работать, но команды Serial будут имитироваться в лог.")