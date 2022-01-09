from numpy.lib.arraysetops import isin
from Copter.TwoMotorStick import TwoMotorsStick
from Copter.Network import Network

import pytest
import torch

def test_init_network():
    net = Network(3, 2)
    model = TwoMotorsStick(net, 0.5, 1e-2)
    return model

def test_compute_angle_acceleration():
    net = Network(3, 2)
    model = TwoMotorsStick(net)
    return model.compute_angle_acceleration(1.)

def test_predict_action_probs():
    net = Network(3, 2)
    model = TwoMotorsStick(net)
    assert isinstance(model.predict_action_probs(), torch.Tensor)
    assert len(model.predict_action_probs()) == 2

def test_compute_angle_acceleration2():
    net = Network(3, 2)
    model = TwoMotorsStick(net)
    model.compute_angle_acceleration(5.)

def test_update_state():
    net = Network(3, 2)
    model = TwoMotorsStick(net)
    model.update_state(5.)
    new_angle = model.state['angle']
    assert new_angle == 0

def test_update_state2():
    net = Network(3, 2)
    model = TwoMotorsStick(net)
    model.update_state(0.)
    model.update_state(0.)
    new_angle = model.state['angle']
    assert new_angle == 0

def test_get_reward():
    net = Network(3, 2)
    model = TwoMotorsStick(net)
    diff = model.update_state(0.)
    r = model.get_reward(diff)
    assert isinstance(r, float)

def test_get_delta_force1():
    net = Network(3, 2)
    model = TwoMotorsStick(net)
    return model.get_delta_force([-30, 830])

def test_get_delta_force2():
    net = Network(3, 2)
    model = TwoMotorsStick(net)
    return model.get_delta_force([512, 830])

def test_step():
    net = Network(3, 2)
    model = TwoMotorsStick(net)
    ans = model.step()
    assert len(ans) == 4

