from model_parameters import *
import numpy as np

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

def compute_acceleration(force_left, force_right, J):
    '''
    Computes rotation acceleration
    Input: force on the left motor, force on the right motor, inertia momentum
    Returns: angle acceleration
    '''
    delta_force = force_right - force_left
    moment = delta_force * MOTOR_DISTANCE / 1000
    return moment * 10e9 / J # in SI

def signal_to_force(self, signal):
    '''
    Empirical function, which shows mapping between signal level given to motor and its resulting force
    In newtons
    '''
    return signal * 10.

def sample_actions(action_probs):
    return np.random.choice(2, 1, p=action_probs)[0]

