from utils import *
from model_parameters import MIN_SIGNAL, MAX_SIGNAL

from typing import List, Tuple, Dict, Type

class TwoMotorsStick(object):
    '''
    An environment for simulations. 
    A model of stick with 2 motors.
    If model_type is binary, every iteration each motor signal is increased or decreased on reaction_speed.
    '''
    def __init__(self, step_size) -> None:
        super().__init__()
        self.J = compute_total_J() # computes inertia moment of model
        self.critical_angle = get_max_angle() * 0.9 # with angle is treated as fail
        self.step_size = step_size
        self.reset()

    def reset(self):
        '''
        Set initial parameters of model
        '''
        # state dict. Is not a state tensor, which is given to network. Have mode complete info about env state.
        self.state = { 
            'angle':0.0,
            'angle_velocity':0.0,
            'angle_acceleration':0.0,
            'angle_jerk':0.0,
        } 

    def get_state(self):
        return self.state

    def compute_angle_acceleration(self, delta_force: float) -> float:
        '''
        Computes actual angle acceleration according to difference in forces
        '''
        return compute_acceleration_using_J(delta_force, self.J)

    def update_state(self, signals: Tuple[int]) -> Dict[str, float]:
        '''
        Computes the differences of parameters and updates the current state
        Input: signals of motors (tuple of 2 integers)
        Returns: difference between new and old states (dict)
        '''
        force_l = signal_to_force(signals[0])
        force_r = signal_to_force(signals[1])
        delta_force = force_r - force_l
        actual_angle_acceleration = self.compute_angle_acceleration(delta_force)
        new_angle = self.state['angle'] + self.state['angle_velocity'] * self.step_size
        new_angle_velocity = self.state['angle_velocity'] + self.state['angle_acceleration'] * self.step_size
        new_jerk = self.state['angle_acceleration'] - actual_angle_acceleration

        feedback = {}
        feedback['delta_angle'] = new_angle - self.state['angle']
        feedback['delta_angle_velocity'] = new_angle_velocity - self.state['angle_velocity']
        feedback['delta_angle_acceleration'] = actual_angle_acceleration - self.state['angle_acceleration']
        feedback['upper_force'] = math.cos(self.state['angle']) * (force_l + force_r)
        min_force, max_force = signal_to_force(MIN_SIGNAL), signal_to_force(MAX_SIGNAL)
        feedback['over_force'] = max(min_force - force_l, 0) + \
            max(min_force - force_l, 0) + \
                max(force_l - max_force, 0) + \
                    max(force_r - max_force, 0)
        feedback['failed'] = abs(new_angle) > self.critical_angle
        
        self.state['angle'] = new_angle
        self.state['angle_velocity'] = new_angle_velocity
        self.state['angle_acceleration'] = actual_angle_acceleration
        self.state['angle_jerk'] = new_jerk
        return feedback
