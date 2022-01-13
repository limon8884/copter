#include <Servo.h>
#include <SPI.h>
#include <Wire.h>
#include <Adafruit_GFX.h>
#include <Adafruit_SSD1306.h>

#define OLED_RESET 4
Adafruit_SSD1306 display(OLED_RESET);

Servo motor;
int motor_pin = 6; // motor 
int voltage_pin = 5;
int voltage_pin2 = 7;
int pot_pin = 0; // potentiomenr

void setup()
{
    pinMode(voltage_pin, OUTPUT);
    digitalWrite(voltage_pin, HIGH);
    pinMode(voltage_pin2, OUTPUT);
    digitalWrite(voltage_pin2, HIGH);
    motor.attach(motor_pin);
    delay(1000);
    display.begin(SSD1306_SWITCHCAPVCC, 0x3C); // display adress
}

void loop()
{
    int val = map(analogRead(pot_pin), 0, 1023, 800, 2300);
    motor.writeMicroseconds(val);
    delay(10);
    display.clearDisplay();
    display.setTextSize(3);
    display.setTextColor(WHITE);
    display.setCursor(8,8);
    display.println(val);
    display.display();
    delay(10);
}
