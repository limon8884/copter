#include "MPU6050.h"
MPU6050 mpu;
int16_t ax, ay, az;
int16_t gx, gy, gz;
const int pot = A0;
int count = 0;
long timer;
const int num_iters = 1000;
//int16_t arr[num_iters][3];

void setup() {
  Wire.begin();
  Serial.begin(9600);
  pinMode(pot, INPUT);
  mpu.initialize();
  mpu.setFullScaleGyroRange(MPU6050_GYRO_FS_500);
  mpu.CalibrateAccel(6);
  mpu.CalibrateGyro(6);
  // состояние соединения
  Serial.println(mpu.testConnection() ? "MPU6050 OK" : "MPU6050 FAIL");
  delay(1000);
}

void loop() {
    if (count < num_iters)
    {   
        if (count == 0) 
        {
            timer = millis();
        }
        mpu.getMotion6(&ax, &ay, &az, &gx, &gy, &gz);
  //  float w = (float)gx / 32768 * 500;
//        mpu.getMotion6(arr[count][1], &ay, arr[count][2], &gx, arr[count][0], &gz);
        Serial.print(ay); Serial.print(',');
        Serial.print(gz); Serial.print(',');
//        Serial.print(az); Serial.print(',');
        
        int pot_val = analogRead(pot);
        Serial.print(pot_val); Serial.print(',');
        Serial.println();
//        Serial.println(count); 
        count++;
        if (count == num_iters)
        {
            double dt = (millis() - timer) / num_iters;
            Serial.println(); 
            Serial.println(count); 
            Serial.println(dt); 
        }
//        Serial.println(count);
//        delay(10);  
    }
    delay(10);
//    Serial.println("kek"); 
}
