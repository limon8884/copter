from Copter.Session import Session
from Copter.Network import Network

import pytest

def test_init_session():
    net = Network(3, 2)
    session = Session(net)
    assert session.success is None

def test_run():
    net = Network(3, 2)
    session = Session(net)
    session.run(10)

def test_get_cumulative_rewards():
    net = Network(3, 2)
    session = Session(net)
    session.run(10)
    ans = session.get_cumulative_rewards()
    assert isinstance(ans, list)
    assert len(ans) == 10

def test_train_model_step():
    net = Network(3, 2)
    session = Session(net)
    session.run(10)
    session.train_model_step()

def test_train_model():
    net = Network(3, 2)
    session = Session(net)
    session.train_model(5)
    


