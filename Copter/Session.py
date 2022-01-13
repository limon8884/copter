from Copter.TwoMotorStick import TwoMotorsStick
from utils import *

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

class Session(object):
    def __init__(self, network, gamma=0.99, jerk_loss_coeff=0.0, step_size=1e-2, std=0.01) -> None:
        super().__init__()
        self.network = network
        self.gamma = gamma # discount factor
        self.jerk_loss_coeff = jerk_loss_coeff
        self.step_size = step_size
        self.std = std
        self.entropy_coef = 0.01

        self.optimizer = torch.optim.Adam(self.network.parameters(), 1e-3)
        self.train_log_rewards = []
        self.reset()
        
    def reset(self):
        '''
        Resets the simulation. Zero all the parameters.
        '''
        self.success = None
        self.model = TwoMotorsStick(self.network, step_size=self.step_size)
        self.iteration = 0
        # self.entropy = math.log(2. * self.model.std**2 * math.pi) + 1 # entropy of 2 variable gaussian
        self.state_history = []
        self.action_history_left = []
        self.action_history_right = []
        self.reward_history = []
        self.out_signals_history = []

    def run(self, n_iters=100, reset=True):
        '''
        Runs session. Simulates a situation for n_iter steps and record all the states and calculates the results
        '''
        if reset:
            self.reset()
        while self.iteration < n_iters:
            self.state_history.append(state_dict_to_tensor(self.model.state))
            reward, action, done, info = self.model.step()
            # self.out_signals_history.append(out_signal)
            self.action_history_left.append(torch.tensor(action[0], dtype=torch.float32))
            self.action_history_right.append(torch.tensor(action[1], dtype=torch.float32))
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

    # def plot_actions(self):
    #     '''
    #     Plots actions (signals on motors) of session
    #     '''
    #     signals_1 = []
    #     signals_2 = []
    #     for s1, s2 in self.action_history:
    #         signals_1.append(s1)
    #         signals_2.append(s2)
    #     plt.plot(signals_1, label='left')
    #     plt.plot(signals_2, label='right')
    #     plt.xlabel('iteration')
    #     plt.ylabel('signal level')
    #     plt.title('Session signals on motors')
    #     plt.legend()
    #     plt.show()
    
    def plot_signals(self):
        signal_tensor = torch.vstack(self.out_signals_history)
        plt.xlabel('iteration')
        plt.ylabel('signals')
        plt.title('Session network output unnormalized')
        plt.plot(signal_tensor[:, 0], label='left')
        plt.plot(signal_tensor[:, 1], label='right')
        plt.legend()
        plt.show()
        
    def plot_states(self):
        '''
        Plots states of session
        '''
        # angle = []
        # velocity = []
        # acceleration = []
        # jerk = []
        # for d in self.state_history:
        #     angle.append(d[0].item())
        #     velocity.append(d[1].item())
        #     acceleration.append(d[2].item())
            # jerk.append(d['jerk'])
        states_tensor = torch.vstack(self.state_history)
        plt.xlabel('iteration')
        plt.ylabel('angle')
        plt.title('session angle')
        plt.plot(states_tensor[:, 0], label='angle')
        plt.show()
        plt.xlabel('iteration')
        plt.ylabel('angle velocity')
        plt.title('Session velocity')
        plt.plot(states_tensor[:, 1], label='velocity')
        plt.show()
        plt.xlabel('iteration')
        plt.ylabel('angle acceleration')
        plt.title('Session acceleration')
        plt.plot(states_tensor[:, 2], label='acceleration')
        plt.show()
        # plt.plot(jerk, label='jerk')
        # plt.legend()
        # plt.show()
    
    def train_model_step(self):
        '''
        Makes a step of model training
        '''
        states_tensor = torch.vstack(self.state_history)
        actions_left_tensor = torch.vstack(self.action_history_left)
        actions_right_tensor = torch.vstack(self.action_history_right)
        cumulative_rewards_tensor = torch.tensor(self.get_cumulative_rewards(), dtype=torch.float32)

        logits = self.network(states_tensor)
        left_log_logits = F.log_softmax(logits[:, :2], -1)
        right_log_logits = F.log_softmax(logits[:, 2:], -1)
        log_probs_for_actions = torch.sum(left_log_logits * to_one_hot(actions_left_tensor, 2) + \
            right_log_logits * to_one_hot(actions_right_tensor, 2), dim=1) 
        
        entropy = (torch.exp(log_probs_for_actions) * log_probs_for_actions).sum()
        loss = -(log_probs_for_actions * cumulative_rewards_tensor).mean() - entropy * self.entropy_coef

        # log_prob = get_log_prob(actions_tensor, preds, self.model.std)
        # loss = -(log_prob * cumulative_rewards_tensor).mean()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return self.model.total_reward

    def train_model(self, train__steps=10, run_iterations=1000):
        '''
        Trains model for several steps
        '''
        for step in range(train__steps):
            self.run(run_iterations)
            reward = self.train_model_step()
            self.train_log_rewards.append(reward)
            print(step, reward, self.iteration)



