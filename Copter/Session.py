from Copter.TwoMotorStick import TwoMotorsStick
from utils import state_dict_to_tensor, get_log_prob

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import math

class Session(object):
    def __init__(self, network, gamma=0.99) -> None:
        super().__init__()
        self.network = network
        self.gamma = gamma # discount factor
        self.optimizer = torch.optim.Adam(self.network.parameters(), 1e-3)
        self.reset()
        
    def reset(self):
        '''
        Resets the simulation. Zero all the parameters.
        '''
        self.success = None
        self.model = TwoMotorsStick(self.network)
        self.iteration = 0
        self.entropy = math.log(2. * self.model.std**2 * math.pi) + 1 # entropy of 2 variable gaussian
        self.state_history = []
        self.action_history = []
        self.reward_history = []

    def run(self, n_iters=100, reset=True):
        '''
        Runs session. Simulates a situation for n_iter steps and record all the states and calculates the results
        '''
        if reset:
            self.reset()
        while self.iteration < n_iters:
            self.state_history.append(state_dict_to_tensor(self.model.state))
            reward, action, done, info = self.model.step()
            self.action_history.append(torch.tensor(action, dtype=torch.float32))
            self.reward_history.append(reward)
            if done:
                self.success = False
                break
            self.iteration += 1
        if self.success is None:
            self.success = True

    def get_cumulative_rewards(self):
        """
        Take a list of immediate rewards r(s,a) for the whole session 
        and compute cumulative returns (a.k.a. G(s,a) in Sutton '16).
        
        G_t = r_t + gamma*r_{t+1} + gamma^2*r_{t+2} + ...

        A simple way to compute cumulative rewards is to iterate from the last
        to the first timestep and compute G_t = r_t + gamma*G_{t+1} recurrently

        You must return an array/list of cumulative rewards with as many elements as in the initial rewards.
        """
        rewards = self.reward_history
        ans = []
        cur = 0
        for i in range(len(rewards) - 1, -1, -1):
            ans.append(cur * self.gamma + rewards[i])
            cur = cur * self.gamma + rewards[i]
        return ans[::-1]

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
            angle.append(d[0].item())
            velocity.append(d[1].item())
            acceleration.append(d[2].item())
            # jerk.append(d['jerk'])
        plt.plot(angle, label='angle')
        plt.plot(velocity, label='velocity')
        plt.plot(acceleration, label='acceleration')
        # plt.plot(jerk, label='jerk')
        plt.xlabel('iteration')
        plt.ylabel('states')
        plt.title('Session states')
        plt.legend()
        plt.show()
    
    def train_model_step(self):
        '''
        Makes a step of model training
        '''
        states_tensor = torch.vstack(self.state_history)
        actions_tensor = torch.vstack(self.action_history)
        cumulative_rewards_tensor = torch.tensor(self.get_cumulative_rewards(), dtype=torch.float32)

        preds = self.network(states_tensor)
        log_prob = get_log_prob(actions_tensor, preds, self.model.std)

        loss = -(log_prob * cumulative_rewards_tensor).mean()
        loss.backward()
        self.optimizer.step()

        return self.model.total_reward

    def train_model(self, num_steps=10):
        '''
        Trains model for several steps
        '''
        for step in range(num_steps):
            self.run()
            reward = self.train_model_step()
            print(reward)



