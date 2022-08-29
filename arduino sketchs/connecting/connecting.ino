// BACKENG FOR ARDUINO MODULE CONNECTED TO INFERENCE

#include "I2Cdev.h"
#include "MPU6050_6Axis_MotionApps20.h"
MPU6050 mpu;

volatile bool mpuFlag = false;  // флаг прерывания готовности
uint8_t fifoBuffer[45];         // буфер
int16_t ax, ay, az;
int16_t gx, gy, gz;

const byte numSignals = 2;
byte signals[numSignals]; // input signals buffer

const float NONE_CONST = 100.0; // fake constant for angle (like None)

const int pot = A0; // potentiometer pin

void setup() {
  Serial.begin(115200);
  pinMode(pot, INPUT);
  Wire.begin();
  //Wire.setClock(1000000UL);     // разгоняем шину на максимум

  mpu.initialize();
  mpu.setFullScaleGyroRange(MPU6050_GYRO_FS_500);
  mpu.CalibrateAccel(6);
  mpu.CalibrateGyro(6);

  // инициализация DMP и прерывания
  mpu.dmpInitialize();
  mpu.setDMPEnabled(true);
  attachInterrupt(0, dmpReady, RISING);
  delay(2000);
}

// прерывание готовности. Поднимаем флаг
void dmpReady() {
  mpuFlag = true;
}

void loop() {
  float current_angle = NONE_CONST;
  // по флагу прерывания и готовности DMP
  if (mpuFlag && mpu.dmpGetCurrentFIFOPacket(fifoBuffer)) {
    // переменные для расчёта (ypr можно вынести в глобал)
    Quaternion q;
    VectorFloat gravity;
    float ypr[3];

    // расчёты
    mpu.dmpGetQuaternion(&q, fifoBuffer);
    mpu.dmpGetGravity(&gravity, &q);
    mpu.dmpGetYawPitchRoll(ypr, &q, &gravity);
    mpuFlag = false;

    current_angle = ypr[0] / 3.1415 * 180;
    delay(15);
  }
  if (Serial.available())
    {    
        // get data from agent
        Serial.readBytes(signals, numSignals);
        // get data from env
        mpu.getMotion6(&ax, &ay, &az, &gx, &gy, &gz);
        int pot_val = analogRead(pot);
        
        // send signals to motors
        // RELEAZE

        String s = String(current_angle) // angle in degrees
            + "," + String(pot_val) // analog signal from angle potentiometer (from 0 to 2023)
            + "," + String((float)gz / 32768 * 500) // angle velocity (radians/s)
            + "," + String((float)ay / 32768 * 2) // acceleration (m/s^2)
            + "," + String(signals[0] + 1) // test thing
        ;
        Serial.print(s);
        delay(20);
    }
   delay(10);
}