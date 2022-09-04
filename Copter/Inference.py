# from sqlite3 import TimeFromTicks
import torch
import serial
import time
import numpy as np

from utils import normalize_tensor

class Inference():
    def __init__(self, network) -> None:
        self.arduino = serial.Serial(port='COM10', baudrate=115200, timeout=0.1)
        self.network = network
        time.sleep(1)
        self.logs = {
            'angle_dmp': [0, 0], # in degrees
            'angle_pot': [0, 0], # in analog signal
            'angle_velocity': [0, 0], # in radians/s
            'acceleration': [0, 0], # raw, with distance and gravity. In m/s^2
            'time': [0, 0], # in seconds
            'signals': [[], []],
            'debug': [0, 0], # signal1 + 1
        }
        self.fuckup_iterations = 0

    def transmit(self, signal, sleep_time=0.01):
        '''
        Input: list of integers from 0 to 255
        '''
        self.arduino.write(bytes(signal))
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

    def go_to_net(self):
        data_in = torch.tensor([
            self.logs['angle_dmp'][-1] / 180 * 3.1415, # angle
            (self.logs['angle_dmp'][-1] - self.logs['angle_dmp'][-2]) / 180 * 3.1415 / self.logs['time'][-1], # velocity
            ((self.logs['angle_dmp'][-1] - self.logs['angle_dmp'][-2]) - (self.logs['angle_dmp'][-2] - self.logs['angle_dmp'][-3]))
                / 180 * 3.1415 / self.logs['time'][-1]**2, # acceleration
            self.logs['signals'][-1][0], # left signal
            self.logs['signals'][-1][1], # right signal
            2.0, # target force
        ], dtype=torch.float)
        data_out = self.network(normalize_tensor(data_in))

        return list(data_out)

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

            self.logs['angle_dmp'].append(data_in[0]) # degrees
            self.logs['angle_pot'].append(data_in[1]) # 0-1024
            self.logs['angle_velocity'].append(data_in[2]) # radians/sec
            self.logs['acceleration'].append(data_in[3]) # m / sec^2
            self.logs['time'].append(time.time() - current_time) # seconds
            self.logs['debug'].append(data_in[4])

            current_time = time.time()
            data_out = self.go_to_net()
            self.logs['signals'].append(data_out)
            data_out = [129, 127]
            self.transmit(data_out)

            if i % 100 == 0:
                print('1 iteration time: ', np.mean(self.logs['time'][-100:]))

        self.arduino.close()

    def stop(self):
        self.arduino.close()

