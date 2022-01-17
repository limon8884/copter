from Copter.TwoMotorStick import TwoMotorsStick
from utils import *

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

class Session(object):
    '''
    This is a model for running simulations and training network in TwoMotorsStick envinronment.
    This have all the logs too.
    '''
    def __init__(self, network, **kwargs) -> None:
        super().__init__()
        self.network = network 
        self.model = TwoMotorsStick(self.network, **kwargs)
        self.gamma = kwargs['gamma'] # discount factor
        self.entropy_coef = 0.01 # for loss functions

        self.optimizer = torch.optim.Adam(self.network.parameters(), 1e-3)
        self.train_log_rewards = [] # log of rewards for all steps (not session iterations)
        self.train_log_iters = []
        self.reset() # resets environment
        
    def reset(self):
        '''
        Resets the simulation environment. Zero all the parameters except network weights.
        '''
        self.success = None
        self.model.reset()
        self.iteration = 0
        # self.entropy = math.log(2. * self.model.std**2 * math.pi) + 1 # entropy of 2 variable gaussian
        self.state_history = []
        self.action_history_left = []
        self.action_history_right = []
        self.reward_history = []
        self.out_signals_history_left = []
        self.out_signals_history_right = []
        self.info_history = []

    def run(self, n_iters=100, reset=True):
        '''
        Runs session. Simulates a situation for n_iter steps and record all the states, actions, rewards.
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
            self.out_signals_history_left.append(self.model.state['left_signal'])
            self.out_signals_history_right.append(self.model.state['right_signal'])
            self.info_history.append(info)
            if done:
                self.success = False
                break
            self.iteration += 1
        if self.success is None:
            self.success = True

    def get_cumulative_rewards(self):
        """
        Computes cumulative rewards for session.

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
        plt.axhline(y=self.model.max_reward, color='r', linestyle='-')
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
        '''
        Plots signals on motors
        '''
        self.out_signals_history_left
        plt.xlabel('iteration')
        plt.ylabel('signals')
        plt.title('Session signals on motors')
        plt.plot(self.out_signals_history_left, label='left')
        plt.plot(self.out_signals_history_right, label='right')
        plt.axhline(y=MAX_SIGNAL, color='r', linestyle='-')
        plt.axhline(y=MIN_SIGNAL, color='r', linestyle='-')
        plt.legend()
        plt.show()
        
    def plot_states(self):
        '''
        Plots states of session
        '''
        states_tensor = torch.vstack(self.state_history)
        plt.xlabel('iteration')
        plt.ylabel('angle')
        plt.title('session angle')
        plt.plot(states_tensor[:, 0], label='angle')
        plt.axhline(y=self.model.critical_angle, color='r', linestyle='-')
        plt.axhline(y=-self.model.critical_angle, color='r', linestyle='-')
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
        max_acc = self.model.compute_angle_acceleration(signal_to_force(MAX_SIGNAL) - signal_to_force(MIN_SIGNAL))
        plt.axhline(y=max_acc, color='r', linestyle='-')
        plt.axhline(y=-max_acc, color='r', linestyle='-')
        plt.show()
        # plt.plot(jerk, label='jerk')
        # plt.legend()
        # plt.show()

    def plot_info(self):
        '''
        Plots info of session
        '''
        info_tensor = torch.tensor(self.info_history)
        plt.xlabel('iteration')
        plt.ylabel('reward/loss')
        plt.title('session info')
        plt.plot(info_tensor[:, 0], label='angle loss')
        plt.plot(info_tensor[:, 1], label='force loss')
        plt.axhline(y=self.model.max_reward, color='r', linestyle='-', label='max reward')
        plt.legend()
        plt.show()

    
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

    def train_model(self, train__steps=10, run_iterations=100, print_=True):
        '''
        Trains model for several steps.
        '''
        for step in range(train__steps):
            self.run(run_iterations)
            reward = self.train_model_step()
            self.train_log_rewards.append(reward)
            self.train_log_iters.append(self.iteration)
            if print_:
                print(step, reward, self.iteration)

    def plot_trained_logs(self, window_size=10):
        r = np.convolve(self.train_log_rewards, np.ones(window_size), 'valid') / window_size
        i = np.convolve(self.train_log_iters, np.ones(window_size), 'valid') / window_size
        plt.xlabel('step')
        plt.ylabel('reward')
        plt.title('rolling reward')
        plt.plot(r)
        plt.show()
        plt.xlabel('step')
        plt.ylabel('iterations')
        plt.title('rolling num of iterations')
        plt.plot(i)
        plt.show()

