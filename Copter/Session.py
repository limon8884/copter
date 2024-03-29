# from pytest import fail
from Copter.TwoMotorStick import TwoMotorsStick
from Copter.Agent import Agent
from utils import *

import time
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

class TargetParamsGenerator(object):
    def __init__(self, pred_session_lenght):
        super().__init__()
        k = 10
        self.pred_session_lenght = pred_session_lenght
        self.precision = 2 / 100
        self.generator_coefs = np.random.randint(-5, 5, k)
        self.init_norm_values()

    def polynom(self, x):
        x_deg = 1
        ans = 0
        for c in self.generator_coefs:
            ans += c * x_deg
            x_deg *= x
        return ans 

    def init_norm_values(self):
        min_board, max_board = self.get_min_max_boards()
        min_force, max_force = signal_to_force(MIN_SIGNAL), signal_to_force(MAX_SIGNAL)

        self.coef = (max_force - min_force) / (max_board - min_board)
        self.intercept = min_force - self.coef * min_board

    def get_min_max_boards(self):
        # min_board = min([self.polynom(x) for x in np.arange(-1, 1, self.precision)])
        # max_board = max([self.polynom(x) for x in np.arange(-1, 1, self.precision)])

        # these are 20 percentile of min_boards and 80 percenltile of max_boards
        return -15, 10
    
    def constrain(self, force):
        min_force, max_force = signal_to_force(MIN_SIGNAL), signal_to_force(MAX_SIGNAL)

        return max(min_force, min(max_force, force))

    def generate_target_params(self, iteration):
        x = 2 * iteration / self.pred_session_lenght - 1
        force = self.intercept + self.coef * self.polynom(x)

        d = {
            'upper_force': self.constrain(force),
        }
        return d


class Session(object):
    '''
    This is a model for running simulations and training network in TwoMotorsStick envinronment.
    This have all the logs too.
    '''
    def __init__(self, network, target_params, **kwargs) -> None:
        super().__init__()
        self.agent = Agent(**kwargs)
        self.env = TwoMotorsStick(**kwargs)

        self.network = network 
        self.target_params = target_params
        self.communication_time_step = kwargs['communication_time_step']

        # self.gamma = kwargs['gamma'] # discount factor
        # self.target_upper_force = kwargs['target_upper_force']
        assert self.network.in_channels == 6
        assert self.network.out_channels == 6

        self.reset() # resets environment
        
    def reset(self):
        '''
        Resets the simulation environment. Zero all the parameters except network weights.
        '''
        self.success = None
        self.env.reset()
        self.agent.reset()
        self.iteration = 0
        self.current_timestamp = 0
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
            'target_upper_force': [],
            'real_upper_force': [],
            'failed': [],
            'action_left': [],
            'action_right': [],
            'network_out_signal': [],
            'time': [],
            'info': []
        }

    def step(self, target_params):
        '''
        On iteration of communication between an agent and the environment
        Returns whether the agent is done
        '''
        state = self.env.get_state()
        self.logs['state_angle'].append(state['angle'])
        self.logs['state_angle_velocity'].append(state['angle_velocity'])
        self.logs['state_angle_acceleration'].append(state['angle_acceleration'])
        self.agent.set_state(state)
        self.agent.set_target_params(target_params)
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
        feedback = self.env.update_state(signals, self.current_timestamp)
        self.agent.get_losses(feedback)
        self.logs['real_upper_force'].append(feedback['upper_force'])
        self.logs['angle_loss'].append(self.agent.losses['angle'])
        self.logs['upper_force_loss'].append(self.agent.losses['upper_force'])
        self.logs['target_upper_force'].append(self.agent.target_params['upper_force'])
        reward = self.agent.get_reward()
        self.logs['reward'].append(reward)
        done = self.agent.is_failed(feedback)
        self.logs['failed'].append(int(done))
        self.total_reward += reward
        self.current_timestamp += self.communication_time_step

        return done

    def run(self, max_iters=100, pred_iters=100, reset=True):
        '''
        Runs session. Simulates a situation for n_iter steps and record all the states, actions, rewards.
        '''
        if reset:
            self.reset()

        target_params_generator = TargetParamsGenerator(pred_iters)
        # start_time = time.time()
        # target_forces = generate_target_forces(n_ticks=n_iters, const_force=self.target_upper_force)
        # gen_time = time.time() - start_time
        # delta_times = []

        while self.iteration < max_iters:
            # start_time = time.time()
            if self.target_params is not None:
                target_params = self.target_params
            else:
                target_params = target_params_generator.generate_target_params(self.iteration)

            failed = self.step(target_params)
            if failed:
                break
            self.iteration += 1
            # delta_times.append(time.time() - start_time)
        self.success = not failed

        # print('gen time: ', gen_time)
        # print('avg step time: ', np.mean(delta_times))

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
            self.logs['target_upper_force'], 
        )), dtype=torch.float).clone().detach()

    def get_action_tensors(self) -> torch.Tensor:
        action_tensor_left = to_one_hot(torch.tensor(self.logs['action_left']), 3).clone().detach()
        action_tensor_right = to_one_hot(torch.tensor(self.logs['action_right']), 3).clone().detach()

        return action_tensor_left, action_tensor_right

    def plot_logs(self):
        '''
        Plots all the logs of current session
        '''
        num_x = 4
        num_y = 3
        fig, axs = plt.subplots(num_y, num_x, sharey=False, figsize=(20, 16))
        # fig.suptitle('Model info')
        for num, (name, arr) in enumerate(self.logs.items()):
            if name in ['network_out_signal', 'info', 'action_left', 'action_right', 'failed', 'target_params', 'time']:
                continue
            ax = axs[num // num_x][num % num_x]
            ax.plot(arr)
            ax.set_title(name, fontsize=15)
 
    
