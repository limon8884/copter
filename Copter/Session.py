# from pytest import fail
from Copter.TwoMotorStick import TwoMotorsStick
from Copter.Agent import Agent
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
        self.agent = Agent(**kwargs)
        self.env = TwoMotorsStick(kwargs['step_size'])
        self.network = network 
        self.gamma = kwargs['gamma'] # discount factor

        self.reset() # resets environment
        
    def reset(self):
        '''
        Resets the simulation environment. Zero all the parameters except network weights.
        '''
        self.success = None
        self.env.reset()
        self.agent.reset()
        self.iteration = 0
        self.total_reward = 0
        # self.entropy = math.log(2. * self.model.std**2 * math.pi) + 1 # entropy of 2 variable gaussian
        self.logs = {
            'state_angle': [],
            'state_angle_velocity': [],
            'state_angle_acceleration': [],
            'signal_left': [],
            'signal_right': [],
            'reward': [],
            'angle_loss': [],
            'upper_force_loss': [],
            'failed': [],
            'action_left': [],
            'action_right': [],
            'network_out_signal': [],
            'info': []
        }

    def step(self):
        '''
        On iteration of communication between an agent and the environment
        Returns whether the agent is done
        '''
        state = self.env.get_state()
        self.logs['state_angle'].append(state['angle'])
        self.logs['state_angle_velocity'].append(state['angle_velocity'])
        self.logs['state_angle_acceleration'].append(state['angle_acceleration'])
        self.agent.set_state(state)
        signal_to_network = normalize_tensor(self.agent.make_signal_to_network())
        with torch.no_grad():
            signal_from_network = self.network(signal_to_network)
        self.agent.get_signal_from_network(signal_from_network)
        self.logs['network_out_signal'].append(signal_from_network)
        signals = self.agent.get_signals()
        self.logs['signal_left'].append(signals[0])
        self.logs['signal_right'].append(signals[1])
        self.logs['action_left'].append(self.agent.actions['left'])
        self.logs['action_right'].append(self.agent.actions['right'])
        feedback = self.env.update_state(signals)
        self.agent.get_losses(feedback)
        self.logs['angle_loss'].append(self.agent.losses['angle'])
        self.logs['upper_force_loss'].append(self.agent.losses['upper_force'])
        reward = self.agent.get_reward()
        self.logs['reward'].append(reward)
        done = self.agent.is_failed(feedback)
        self.logs['failed'].append(int(done))
        self.total_reward += reward

        return done

    def run(self, n_iters=100, reset=True):
        '''
        Runs session. Simulates a situation for n_iter steps and record all the states, actions, rewards.
        '''
        if reset:
            self.reset()
        while self.iteration < n_iters:
            failed = self.step()
            if failed:
                break
            self.iteration += 1
        self.success = not failed

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
        rewards = self.logs['reward']
        ans = []
        cur = 0
        for i in range(len(rewards) - 1, -1, -1):
            ans.append(cur * self.gamma + rewards[i])
            cur = cur * self.gamma + rewards[i]
        return ans[::-1]

    def get_state_tensor(self) -> torch.Tensor:
        return torch.tensor(list(zip(
            self.logs['state_angle'],
            self.logs['state_angle_velocity'],
            self.logs['state_angle_acceleration'],
            self.logs['signal_left'],
            self.logs['signal_right'],
        )), dtype=torch.float).clone().detach()

    def get_action_tensors(self) -> torch.Tensor:
        action_tensor_left = to_one_hot(torch.tensor(self.logs['action_left']), 2).clone().detach()
        action_tensor_right = to_one_hot(torch.tensor(self.logs['action_right']), 2).clone().detach()

        return action_tensor_left, action_tensor_right

    def plot_logs(self):
        '''
        Plots all the logs of current session
        '''
        num_x = 3
        num_y = 3
        fig, axs = plt.subplots(num_y, num_x, sharey=False, figsize=(20, 16))
        # fig.suptitle('Model info')
        for num, (name, arr) in enumerate(self.logs.items()):
            if name in ['network_out_signal', 'info', 'action_left', 'action_right']:
                continue
            ax = axs[num // num_x][num % num_x]
            ax.plot(arr)
            ax.set_title(name, fontsize=15)
 
    
