from numpy.lib.arraysetops import isin
from Copter.TwoMotorStick import TwoMotorsStick

import pytest
import torch

def test_init_network():
    model = TwoMotorsStick(0.5, 1e-2)
    return model

def test_compute_angle_acceleration():
    model = TwoMotorsStick()
    return model.compute_angle_acceleration(1.)

def test_predict_action_probs():
    model = TwoMotorsStick()
    assert isinstance(model.predict_action_probs(), list)
    assert len(model.predict_action_probs()) == 4

def test_compute_angle_acceleration():
    model = TwoMotorsStick()
    model.compute_angle_acceleration(5.)

def test_update_state():
    model = TwoMotorsStick()
    model.update_state(5.)
    new_angle = model.state['angle']
    assert new_angle == 0

def test_update_state2():
    model = TwoMotorsStick()
    model.update_state(0.)
    model.update_state(0.)
    new_angle = model.state['angle']
    assert new_angle == 0

def test_get_reward():
    model = TwoMotorsStick()
    diff = model.update_state(0.)
    r = model.get_reward(diff)
    assert isinstance(r, float)

def test_get_delta_force1():
    model = TwoMotorsStick()
    return model.get_delta_force([-30, 830])

def test_get_delta_force2():
    model = TwoMotorsStick()
    return model.get_delta_force([512, 830])

def test_step():
    model = TwoMotorsStick()
    ans = model.step()
    assert len(ans) == 4

