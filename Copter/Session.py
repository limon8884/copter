from TwoMotorStick import TwoMotorsStick

import matplotlib.pyplot as plt

class Session(object):
    def __init__(self) -> None:
        super().__init__()
        self.reset()
        
    def reset(self):
        '''
        Resets the simulation. Zero all the parameters.
        '''
        self.success = None
        self.model = TwoMotorsStick()
        self.iteration = 0
        self.state_history = []
        self.action_history = []
        self.reward_history = []

    def run(self, n_iters=1000, reset=True):
        '''
        Runs session. Simulates a situation for n_iter steps and record all the states and calculates the results
        '''
        if reset:
            self.reset()
        while self.iteration < n_iters:
            self.state_history.append(self.model.state)
            reward, action, done, info = self.model.step()
            self.action_history.append(action)
            self.reward_history.append(reward)
            if done:
                self.success = False
                break
            self.iteration += 1
        if self.success is None:
            self.success = True

    def plot_histories(self, one_plot=True):
        '''
        Plots all histories of session
        '''
        pass