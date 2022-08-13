from model_parameters import *
from model_parameters import SIGNAL_COEF

import numpy as np
import math
import torch
import torch.nn.functional as F
from typing import List, Tuple, Type, Dict

def compute_total_J(print_=False) -> float:
    '''
    Computes inertia momentum of system according to model parameters
    '''
    J_motor = 2 * MOTOR_MASS * (MOTOR_DISTANCE**2 + (MOTOR_HEIGHT - CENTER_AXIS_IDENT)**2)
    J_reg = 2 * REGULATOR_MASS * (REGULATOR_DISTANCE**2 + (REGULATOR_HEIGHT - CENTER_AXIS_IDENT)**2)
    J_stick = STICK_MASS * ((STICK_LENGTH**2 + STICK_HEIGHT**2) / 3. + CENTER_AXIS_IDENT**2)
    if print_:
        print(J_motor, J_reg, J_stick) 
    return J_motor + J_reg + J_stick

def compute_acceleration_using_J(delta_force: float, J: float) -> float:
    '''
    Computes rotation acceleration
    Input: difference of forces on the right and left motors, inertia momentum
    Returns: angle acceleration
    '''
    moment = delta_force * MOTOR_DISTANCE
    return moment / J 

def network_output_to_signal(output):
    '''
    Transforms output of network (from [-1, 1]) to signal range (from MIN_SIGNAL to MAX_SIGNAL) 
    '''
    if output.item() < -1:
        output = torch.tensor(-1)
    if output.item() > 1:
        output = torch.tensor(1)
    return (output + 1.) * (MAX_SIGNAL - MIN_SIGNAL) / 2. + MIN_SIGNAL

def signal_to_force(signal: float, std_force=0.0) -> float:
    '''
    Empirical function, which shows mapping between signal level given to motor and its resulting force
    In newtons
    '''
    # constrained_signal = SIGNAL_TYPE - F.relu(SIGNAL_TYPE - F.relu(torch.tensor(signal)))
    # normalised_signal = 2. * constrained_signal / SIGNAL_TYPE - 1.
    # normalised_force = torch.tanh(normalised_signal * SCALE_FACTOR)
    # min_value, max_value = torch.tanh(torch.tensor(-SCALE_FACTOR)), torch.tanh(torch.tensor(SCALE_FACTOR))
    # real_force = (normalised_force - min_value) / (max_value - min_value) * MAX_FORCE
    if signal < MIN_SIGNAL:
        signal = MIN_SIGNAL
    if signal > MAX_SIGNAL:
        signal = MIN_SIGNAL
    noise = np.random.randn() * std_force
    return signal * SIGNAL_COEF + SIGNAL_INTERCEPT + noise


def sample_actions(action_probs_l: np.ndarray, action_probs_r: np.ndarray) -> Tuple[int]:
    '''
    Sample signals on motors.
    Input: list of np.arrays. Distribution parameters. 
    In case of continious model these parameters are means for both motors (torch.tensor) and their var (float)
    In case of binary model these are 2 Bernulli distributions (2 numpy arrays)
    Returns: 2 values of signals (list of tensors)
    '''
    # if binary:
    a_l = np.random.choice(2, 1, p=action_probs_l)[0]
    a_r = np.random.choice(2, 1, p=action_probs_r)[0]
    return a_l, a_r
    # else:
    #     mean_l, mean_r, std = args
    #     return mean_l + torch.randn(1) * std, mean_r + torch.randn(1) * std
    # N_l = torch.distributions.normal.Normal(loc=mean_l, scale=torch.tensor(std, requires_grad=False, dtype=torch.float32))
    # N_r = torch.distributions.normal.Normal(loc=mean_r, scale=torch.tensor(std, requires_grad=False, dtype=torch.float32))  u are in THE guy) true
    # return N_l.rsample(), N_r.rsample()

def add_noise(tens, noise_std=0.0):
    '''
    Adds noise to tensor
    '''
    return tens + torch.randn(tens.shape) * noise_std

def get_max_angle():
    '''
    Returns max angle in radians, in which the game is done
    '''
    return MAX_ANGLE / 180. * math.pi

def get_log_prob(actions, preds, std):
    log_probs = -torch.square(preds - actions) / 2 - math.log((2. * math.pi)**0.5 / std)
    return log_probs.sum(dim=-1)

def to_one_hot(y_tensor: torch.Tensor, ndims: int) -> torch.Tensor:
    """ helper: take an integer vector and convert it to 1-hot matrix. """
    y_tensor = y_tensor.type(torch.LongTensor).view(-1, 1)
    y_one_hot = torch.zeros(y_tensor.size()[0], ndims).scatter_(1, y_tensor, 1)
    return y_one_hot

def normalize_tensor(input: torch.Tensor, mean=None, scale=None) -> torch.Tensor:
    '''
    Normalize data for NN. Substract mean then divide by scale every sample of input tensor.
    Input: 
    - tensor of shape (n_samples, in_channels) where n_channels corresponds to number of input channels of NN
    - tensor of shape (in_channels,)
    - tensor of shape (in_channels,)
    Returns: tensor of shape (n_samples, in_channels) 
    '''
    num_channels = input.shape[-1]
    if mean is None:
        # if num_channels == 5:
            # mean = torch.tensor([0, 0, 0, (MAX_SIGNAL - MIN_SIGNAL) / 2, (MAX_SIGNAL - MIN_SIGNAL) / 2])
        # else:
        min_force, max_force = signal_to_force(MIN_SIGNAL), signal_to_force(MAX_SIGNAL)
        mean = torch.tensor([0, 0, 0, (MAX_SIGNAL - MIN_SIGNAL) / 2, (MAX_SIGNAL - MIN_SIGNAL) / 2, (max_force - min_force) / 2])
    if scale is None:
        max_ang = MAX_ANGLE * 0.9 / 180 * 3.1415
        max_acc = 30
        # if num_channels == 5:
        #     scale = torch.tensor([max_ang, max_acc * 1e-3, max_acc, (MAX_SIGNAL - MIN_SIGNAL) / 2, (MAX_SIGNAL - MIN_SIGNAL) / 2])
        # else:
        scale = torch.tensor([max_ang, max_acc * 1e-3, max_acc, (MAX_SIGNAL - MIN_SIGNAL) / 2, (MAX_SIGNAL - MIN_SIGNAL) / 2, (max_force - min_force) / 2])
    output = (input - mean) / scale
    assert input.shape == output.shape
    return output

def generate_target_forces(n_ticks: int, const_force: float) -> List[float]:
    if const_force is not None:
        return [const_force] * n_ticks
    k = 10
    coefs = np.random.randint(-5, 5, k)
    step_interval = 2 / n_ticks
    def polynom(x):
        x_deg = 1
        ans = 0
        for c in coefs:
            ans += c * x_deg
            x_deg *= x
        return ans

    arr = [polynom(x) for x in np.arange(-1, 1, step_interval)]
    min_force, max_force = signal_to_force(MIN_SIGNAL), signal_to_force(MAX_SIGNAL)
    coef = (max_force - min_force) / (max(arr) - min(arr))
    min_a = min(arr)
    forces = [min_force + coef * (a - min_a) for a in arr]

    return forces
