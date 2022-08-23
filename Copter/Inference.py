from sqlite3 import TimeFromTicks
import serial
import time



class Inference():
    def __init__(self, network) -> None:
        self.arduino = serial.Serial(port='COM10', baudrate=115200, timeout=.1)
        self.network = network

    def transmit(self, signal):
        self.arduino.write(bytes(signal, 'utf-8')) # надо бы флаты в байты обернуть

    def recieve(self):
        return self.arduino.readline()

    def run(self):
        current_time = 0
        for i in range(10000):
            current_time += time.time()
            data_in = self.recieve() # надо бы кастануть
            data_out = self.network(data_in)
            time.sleep(0.001)
            self.transmit(data_out)
            self.arduino.flush() # здесь ли?

            if i % 100 == 0:
                print('1 iteration time: ', (time.time() - current_time) / 100)
                current_time = 0

