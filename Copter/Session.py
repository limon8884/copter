from Copter.TwoMotorStick import TwoMotorsStick

import matplotlib.pyplot as plt

from utils import get_log_prob_and_entropy

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

    def plot_rewards(self):
        '''
        Plots rewards of session
        '''
        plt.plot(self.reward_history, label='rewards')
        plt.xlabel('iteration')
        plt.ylabel('reward')
        plt.title('Session rewards')
        plt.show()

    def plot_actions(self):
        '''
        Plots actions (signals on motors) of session
        '''
        signals_1 = []
        signals_2 = []
        for s1, s2 in self.action_history:
            signals_1.append(s1)
            signals_2.append(s2)
        plt.plot(signals_1, label='left')
        plt.plot(signals_2, label='right')
        plt.xlabel('iteration')
        plt.ylabel('signal level')
        plt.title('Session signals on motors')
        plt.legend()
        plt.show()
        
    def plot_states(self):
        '''
        Plots states of session
        '''
        angle = []
        velocity = []
        acceleration = []
        jerk = []
        for d in self.state_history:
            angle.append(d['angle'])
            velocity.append(d['angle_velocity'])
            acceleration.append(d['angle_acceleration'])
            jerk.append(d['jerk'])
        plt.plot(angle, label='angle')
        plt.plot(velocity, label='velocity')
        plt.plot(acceleration, label='acceleration')
        plt.plot(jerk, label='jerk')
        plt.xlabel('iteration')
        plt.ylabel('states')
        plt.title('Session states')
        plt.legend()
        plt.show()
    
    # def train_model(self):
    #     states_tensor = 
    #     actions_tensor = 
    #     cumulative_rewards_tensor = 
    #     log_prob, entropy = get_log_prob_and_entropy()

