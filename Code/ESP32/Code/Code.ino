#include <AccelStepper.h>

// ==========================================
// ⚙️ ОСНОВНЫЕ НАСТРОЙКИ
// ==========================================
const int STEPS_PER_REV = 200;  // Шагов на 1 оборот двигателя
const int MICROSTEPS = 4;       // Микрошаг на драйвере
const float STEPS_PER_REVOLUTION = STEPS_PER_REV * MICROSTEPS; // 800 шагов на оборот

// Пины управления моторами и концевиками
const int dirPin1 = 22; const int stepPin1 = 23; const int enPin1 = 16; const int limitPin1 = 21;
const int dirPin2 = 18; const int stepPin2 = 19; const int enPin2 = 2;  const int limitPin2 = 15;

// ==========================================
// 🛑 ПЕРЕМЕННЫЕ СОСТОЯНИЯ
// ==========================================
AccelStepper motor1(AccelStepper::DRIVER, stepPin1, dirPin1);
AccelStepper motor2(AccelStepper::DRIVER, stepPin2, dirPin2);

String inputBuffer = "";
bool isMoving = false;
bool sendStepSignalOnFinish = false; // Флаг, нужно ли отправлять motor:step при остановке

#define MAX_SPEED 800
#define MAX_ACCEL 900

void setup() {
  Serial.begin(115200);
  
  pinMode(enPin1, OUTPUT); digitalWrite(enPin1, LOW);
  pinMode(enPin2, OUTPUT); digitalWrite(enPin2, LOW);
  pinMode(limitPin1, INPUT);
  pinMode(limitPin2, INPUT);

  // Настройка скорости и ускорения моторов
  motor1.setMaxSpeed(MAX_SPEED);     
  motor1.setAcceleration(MAX_ACCEL);  
  motor2.setMaxSpeed(MAX_SPEED);
  motor2.setAcceleration(MAX_ACCEL);

  // calibrate();
}

void loop() {
  // Остановка по достижении дистанции
  if (isMoving && motor1.distanceToGo() == 0 && motor2.distanceToGo() == 0) {
    stopMotors();
    if (sendStepSignalOnFinish) {
      Serial.println("motor:step"); // Отправка обратно в Python после прохождения всех оборотов
      sendStepSignalOnFinish = false;
    }
  }

  // Чтение команд из Serial
  while (Serial.available() > 0) {
    char c = Serial.read();
    if (c == '\n' || c == '\r') {
      if (inputBuffer.length() > 0) {
        processCommand(inputBuffer);
        inputBuffer = "";
      }
    } else { 
      inputBuffer += c;
    }
  }
  
  // Вращение моторов
  motor1.run();
  motor2.run();
}

void stopMotors() {
  motor1.setSpeed(0); motor1.moveTo(motor1.currentPosition());
  motor2.setSpeed(0); motor2.moveTo(motor2.currentPosition());
  isMoving = false;
}

void processCommand(String command) {
  command.trim();
  
  if (command == "move:stop") {
    sendStepSignalOnFinish = false;
    stopMotors();
    return;
  }
  
  // Команда START_UP:число оборотов
  if (command.startsWith("START_UP:")) {
    float totalRevolutions = command.substring(9).toFloat();
    long totalSteps = round(totalRevolutions * STEPS_PER_REVOLUTION);
    
    sendStepSignalOnFinish = true; // Нужен сигнал motor:step в конце
    
    // Вращение в разные стороны
    motor1.move(-totalSteps);
    motor2.move(totalSteps);
    
    isMoving = true;
    return;
  }
 
  // Команда START_DOWN:число оборотов
  if (command.startsWith("START_DOWN:")) {
    float totalRevolutions = command.substring(11).toFloat();
    long totalSteps = round(totalRevolutions * STEPS_PER_REVOLUTION);
    
    sendStepSignalOnFinish = false; // Без обратной связи
    
    // Вращение в противоположные стороны относительно START_UP
    motor1.move(totalSteps);
    motor2.move(-totalSteps);
    
    isMoving = true;
    return;
  }

  // Команда калибровки
  if (command == "START_CALIBRATE" || command == "CALIBRATE") {
    sendStepSignalOnFinish = false;
    calibrate();
    return;
  }

    if (command.startsWith("ABS_MOVE:")) {
    // диапазон 0..5000
    long targetPos = command.substring(9).toInt();
    
    motor1.moveTo(-targetPos); // Устанавливаем абсолютную целевую позицию
    motor2.moveTo(targetPos); // Если нужно двигать оба мотора в одну абсолютную точку
    
    isMoving = true;
    sendStepSignalOnFinish = true; // Чтобы отправить сигнал по завершении
    return;
  }

    if (command.startsWith("SET_SPEED:")) {
    float newSpeed = command.substring(10).toFloat();
    if (newSpeed > 0) {
      motor1.setMaxSpeed(newSpeed);
      motor2.setMaxSpeed(newSpeed);
      Serial.print("speed:set:");
      Serial.println(newSpeed);
    }
    return;
  }

  if (command.startsWith("SET_ACCEL:")) {
    float newAccel = command.substring(10).toFloat();
    if (newAccel > 0) {
      motor1.setAcceleration(newAccel);
      motor2.setAcceleration(newAccel);
      Serial.print("accel:set:");
      Serial.println(newAccel);
    }
    return;
  }
}

void calibrate() {
  stopMotors();

  // 1. Поднимаем моторы на 1 оборот в разные стороны

  if (digitalRead(limitPin1) == LOW || digitalRead(limitPin2) == LOW)
  {
    motor1.move(-STEPS_PER_REVOLUTION);
    motor2.move(STEPS_PER_REVOLUTION);
    
    while (motor1.distanceToGo() != 0 || motor2.distanceToGo() != 0) {
      motor1.run();
      motor2.run();
    }
  }

  // 2. Опускаемся вниз до нажатия каждого концевика
  motor1.setSpeed(400);  
  motor2.setSpeed(-400); 

  bool m1Calibrated = false;
  bool m2Calibrated = false;

  while (!m1Calibrated || !m2Calibrated) {
    if (!m1Calibrated) {
      if (digitalRead(limitPin1) == LOW) {
        motor1.setSpeed(0);
        motor1.setCurrentPosition(0);
        m1Calibrated = true;
      } else {
        motor1.runSpeed();
      }
    }


    if (!m2Calibrated) {
      if (digitalRead(limitPin2) == LOW) {
        motor2.setSpeed(0);
        motor2.setCurrentPosition(0);
        m2Calibrated = true;
      } else {
        motor2.runSpeed();
      }
    }
  }

  // Сброс настроек целевых позиций после калибровки
  stopMotors();
  motor1.setMaxSpeed(MAX_SPEED); motor1.setAcceleration(MAX_ACCEL);
  motor2.setMaxSpeed(MAX_SPEED); motor2.setAcceleration(MAX_ACCEL);
}
