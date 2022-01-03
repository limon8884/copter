from model_parameters import *

def compute_total_J(print_=False):
    J_motor = 2 * MOTOR_MASS * (MOTOR_DISTANCE**2 + (MOTOR_HEIGHT - CENTER_AXIS_IDENT)**2)
    J_reg = 2 * REGULATOR_MASS * (REGULATOR_DISTANCE**2 + (REGULATOR_HEIGHT - CENTER_AXIS_IDENT)**2)
    J_stick = STICK_MASS * ((STICK_LENGTH**2 + STICK_HEIGHT**2) / 3. + CENTER_AXIS_IDENT**2)
    if print_:
        print(J_motor, J_reg, J_stick) 
    return J_motor + J_reg + J_stick

def compute_acceleration(force_left, force_right, J):
    delta_force = force_right - force_left
    moment = delta_force * MOTOR_DISTANCE
    return moment / J 

