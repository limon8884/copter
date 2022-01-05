from model_parameters import *

import numpy as np
import math
import torch
import torch.nn.functional as F

def compute_total_J(print_=False):
    '''
    Computes inertia momentum of system according to model parameters
    '''
    J_motor = 2 * MOTOR_MASS * (MOTOR_DISTANCE**2 + (MOTOR_HEIGHT - CENTER_AXIS_IDENT)**2)
    J_reg = 2 * REGULATOR_MASS * (REGULATOR_DISTANCE**2 + (REGULATOR_HEIGHT - CENTER_AXIS_IDENT)**2)
    J_stick = STICK_MASS * ((STICK_LENGTH**2 + STICK_HEIGHT**2) / 3. + CENTER_AXIS_IDENT**2)
    if print_:
        print(J_motor, J_reg, J_stick) 
    return J_motor + J_reg + J_stick

def compute_acceleration(delta_force, J):
    '''
    Computes rotation acceleration
    Input: difference of forces on the right and left motors, inertia momentum
    Returns: angle acceleration
    '''
    moment = delta_force * MOTOR_DISTANCE * 1e-3
    return moment * 10e9 / J # in SI

def signal_to_force(self, signal):
    '''
    Empirical function, which shows mapping between signal level given to motor and its resulting force
    In newtons
    '''
    assert isinstance(signal, torch.tensor), 'signal type should be a tensor'
    scale_factor = 2. 
    signal_type = 1024. # for 10 bit signal type
    max_force = 10. # in newtons

    constrained_signal = signal_type - F.relu(signal_type - F.relu(torch.tensor(signal)))
    normalised_signal = 2. * constrained_signal / signal_type - 1.
    normalised_force = F.tanh(normalised_signal * scale_factor)
    min_value, max_value = F.tanh(torch.tensor(-scale_factor)), F.tanh(torch.tensor(scale_factor))
    real_force = normalised_force - min_value / (max_value - min_value) * max_force
    return real_force


def sample_actions(action_distribution_parameters):
    '''
    Sample signals on motors.
    Input: list with distribution parameters. In case of normal distribution these parameters are mean and var for both motors
    Returns: 2 values of signals
    '''
    mean_l, var_l, mean_r, var_r = action_distribution_parameters
    return np.random.normal(mean_l, var_l), np.random.normal(mean_r, var_r)

def state_dict_to_tensor(state_dict):
    assert isinstance(state_dict, dict)
    ans_list = [state_dict['angle'], state_dict['angle_velocity'], state_dict['angle_acceleration']]
    return torch.tensor(ans_list).float()

def get_max_angle():
    '''
    Returns max angle in radians, in which the game is done
    '''
    return MAX_ANGLE / 180. * 2. * math.pi

def get_cumulative_rewards(rewards,  # rewards at each step
                           gamma=0.99  # discount for reward
                           ):
    """
    Take a list of immediate rewards r(s,a) for the whole session 
    and compute cumulative returns (a.k.a. G(s,a) in Sutton '16).
    
    G_t = r_t + gamma*r_{t+1} + gamma^2*r_{t+2} + ...

    A simple way to compute cumulative rewards is to iterate from the last
    to the first timestep and compute G_t = r_t + gamma*G_{t+1} recurrently

    You must return an array/list of cumulative rewards with as many elements as in the initial rewards.
    """
    ans = []
    cur = 0
    for i in range(len(rewards) - 1, -1, -1):
      ans.append(cur * gamma + rewards[i])
      cur = cur * gamma + rewards[i]
    return ans[::-1]

