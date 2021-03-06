import pytest
import torch

from utils import *

def test_compute_total_J():
    compute_total_J(True)

def test_compute_acceleration_using_J_1():
    compute_acceleration_using_J(5, compute_total_J())

def test_compute_acceleration_using_J_2():
    assert compute_acceleration_using_J(0, compute_total_J()) == 0

def test_compute_acceleration_using_J_3():
    return compute_acceleration_using_J(-5, compute_total_J())

def test_signal_to_force():
    a = signal_to_force(torch.tensor(1000))

def test_sample_actions():
    # a, b = sample_actions(False, torch.tensor(1, dtype=torch.float), torch.tensor(1, dtype=torch.float), 1)
    # a, b = sample_actions(False, torch.tensor(-1, dtype=torch.float), torch.tensor(1, dtype=torch.float), 1)
    a, b = sample_actions(np.array([0.3, 0.7]), np.array([0.3, 0.7]))

def test_state_dict_to_tensor():
    d = {
        'angle':-1.,
            'angle_velocity':0.,
            'angle_acceleration':1.,
            'angle_jerk':5.,
    }
    tens = state_dict_to_tensor(d)
    assert tens.shape[0] == 3

def test_get_log_prob():
    a = torch.tensor([[1, 2], [3, 4], [5, 6]], dtype=torch.float)
    b = torch.tensor([[-1, 2], [-3, 4], [5, -6]], dtype=torch.float)
    get_log_prob(a, b, 1.)

# def test_network_output_to_signal():
#     a = torch.tensor([1, 2], dtype=torch.float)
#     network_output_to_signal(a)