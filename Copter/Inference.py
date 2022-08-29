# from sqlite3 import TimeFromTicks
import torch
import serial
import time
import numpy as np

class Inference():
    def __init__(self, network) -> None:
        self.arduino = serial.Serial(port='COM10', baudrate=115200, timeout=0.1)
        self.network = network
        time.sleep(1)
        self.logs = {
            'angle_dmp': [], # in degrees
            'angle_pot': [], # in analog signal
            'angle_velocity': [], # in radians/s
            'acceleration': [], # raw, with distance and gravity. In m/s^2
            'time': [], # in seconds
            'debug': [], # signal1 + 1
        }
        self.fuckup_iterations = 0

    def transmit(self, signal, sleep_time=0.01):
        '''
        Input: list of integers from 0 to 255
        '''
        self.arduino.write(bytes(list(signal)))
        time.sleep(sleep_time)

    def recieve(self):
        if self.arduino.in_waiting > 0:
            s = self.arduino.readline(self.arduino.in_waiting).decode('utf-8')
            vals = list(map(float, s.split(',')))
            if len(vals) != 5:
                self.fuckup_iterations += 1
                return [None] * 5

            return vals
        return [None] * 5

    def flush(self, delay=0.01):
        self.arduino.readline(self.arduino.in_waiting)
        self.transmit([0, 0])
        time.sleep(delay)

    def run(self, n_iters=1000):
        self.flush(delay=5)
        for _ in range(5):
            self.flush()
            
        current_time = time.time()
        for i in range(n_iters):
            while self.arduino.in_waiting == 0:
                pass
            data_in = self.recieve() 


            self.logs['angle_dmp'].append(data_in[0])
            self.logs['angle_pot'].append(data_in[1])
            self.logs['angle_velocity'].append(data_in[2])
            self.logs['acceleration'].append(data_in[3])
            self.logs['time'].append(time.time() - current_time)
            self.logs['debug'].append(data_in[4])

            current_time = time.time()
            # data_out = self.network(data_in)
            data_out = [129, 127]
            self.transmit(data_out)

            if i % 100 == 0:
                # print(data_in)
                print('1 iteration time: ', np.mean(self.logs['time'][-100:]))
                # current_time = time.time()

        self.arduino.close()

    def stop(self):
        self.arduino.close()

