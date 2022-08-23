import serial
import time

arduino = serial.Serial(port='COM10', baudrate=115200, timeout=.1)
def write_read(x):
    arduino.write(bytes(x, 'utf-8'))
    time.sleep(0.001)
    data = arduino.readline()
    return data
while True:
    num = input("Enter a number: ") # Taking input from user
    start_time = time.time()
    value = write_read(num)
    print(value)
    print('time: ', time.time() - start_time)