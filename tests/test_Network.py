from Copter.Network import Network

import pytest
import torch

def test_init_network():
    net = Network(5, 5, 5, 0)
    return net

def test_forward():
    net = Network(2, 5, 5, 0)
    input = torch.tensor([[1, 2]], dtype=torch.float32)
    return net(input)

