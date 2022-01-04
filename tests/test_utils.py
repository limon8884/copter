import pytest
from utils import compute_total_J, compute_acceleration

def test_compute_total_J():
    compute_total_J(True)

def test_compute_acceleration1():
    compute_acceleration(5., 10., compute_total_J())

def test_compute_acceleration2():
    assert compute_acceleration(5., 5., compute_total_J()) == 0