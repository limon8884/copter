# from sqlite3 import TimeFromTicks
import torch
import serial
import time

class Inference():
    def __init__(self, network) -> None:
        self.arduino = serial.Serial(port='COM10', baudrate=115200, timeout=0.1)
        self.network = network
        time.sleep(1)

    def transmit(self, signal, sleep_time=0.01):
        '''
        Input: list of integers from 0 to 255
        '''
        self.arduino.write(bytes(list(signal)))
        time.sleep(sleep_time)

    def recieve(self):
        if self.arduino.in_waiting > 0:
            s = self.arduino.readline(self.arduino.in_waiting).decode('utf-8')
            vals = list(map(int, s.split(',')))
            assert len(vals) == 3, vals
            return torch.tensor(vals)
        return None

    def run(self, n_iters=1000):
        time.sleep(2)
        self.arduino.readline(self.arduino.in_waiting)
        self.transmit([0, 0])
        current_time = time.time()
        for i in range(n_iters):
            # print('iter: ', i)
            while self.arduino.in_waiting == 0:
                pass
            # current_time += time.time()
            data_in = self.recieve() 
            # data_out = self.network(data_in)
            data_out = [129, 127]
            self.transmit(data_out)

            if i % 100 == 0:
                print('1 iteration time: ', (time.time() - current_time) / 100)
                current_time = time.time()

        self.arduino.close()

    def stop(self):
        self.arduino.close()

