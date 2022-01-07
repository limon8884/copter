from Copter.Session import Session

import pytest

def test_init_session():
    session = Session()
    assert session.success is None

def test_run():
    session = Session()
    session.run(10)
    


