from turtle import st
from utils import *
from model_parameters import MIN_SIGNAL, MAX_SIGNAL

from typing import List, Tuple, Dict, Type

class TwoMotorsStick(object):
    '''
    An environment for simulations. 
    A model of stick with 2 motors.
    If model_type is binary, every iteration each motor signal is increased or decreased on reaction_speed.
    '''
    def __init__(self, **kwargs) -> None:
        super().__init__()
        self.J = compute_total_J() # computes inertia moment of model
        self.critical_angle = get_max_angle() * 0.9 # with angle is treated as fail

        self.env_update_time_step = kwargs['env_update_time_step']

        self.std_angle = kwargs['std_angle']
        self.std_velocity = kwargs['std_velocity']
        self.std_acceleration = kwargs['std_acceleration']
        self.std_force = kwargs['std_force']

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
        self.signals = {
            'left': MIN_SIGNAL,
            'right': MIN_SIGNAL,
        }
        self.last_updated_ts = None

    def get_state(self):
        state = {}
        state['angle'] = self.state['angle'] + np.random.randn() * self.std_angle
        state['angle_velocity'] = self.state['angle_velocity'] + np.random.randn() * self.std_velocity
        state['angle_acceleration'] = self.state['angle_acceleration'] + np.random.randn() * self.std_acceleration

        return state

    def compute_angle_acceleration(self, delta_force: float) -> float:
        '''
        Computes actual angle acceleration according to difference in forces
        '''
        return compute_acceleration_using_J(delta_force, self.J)

    def next_state(self):
        '''
        Simulates the motion of the env after the small time step
        '''
        right_force = signal_to_force(self.signals['right'], self.std_force)
        left_force = signal_to_force(self.signals['left'], self.std_force)

        actual_angle_acceleration = self.compute_angle_acceleration(right_force - left_force)

        new_angle = self.state['angle'] + self.state['angle_velocity'] * self.env_update_time_step
        new_angle_velocity = self.state['angle_velocity'] + self.state['angle_acceleration'] * self.env_update_time_step
        new_jerk = self.state['angle_acceleration'] - actual_angle_acceleration

        self.state['angle'] = new_angle
        self.state['angle_velocity'] = new_angle_velocity
        self.state['angle_acceleration'] = actual_angle_acceleration
        self.state['angle_jerk'] = new_jerk

    def update_state(self, signals: Tuple[int], timestamp: float) -> Dict[str, float]:
        '''
        Simulates behavior of env since the last call of this function 
        Input: signals of motors (tuple of 2 integers)
        Returns: current state of env (dict)
        '''
        if self.last_updated_ts is None:
            n_steps = 0
        else:
            n_steps = int( (timestamp - self.last_updated_ts) / self.env_update_time_step )
        self.last_updated_ts = timestamp

        for _ in range(n_steps):
            self.next_state()

        feedback = self.get_state().copy()
        feedback['upper_force'] = \
            math.cos(self.state['angle']) * (signal_to_force(self.signals['left']) + signal_to_force(self.signals['right']))
        feedback['failed'] = abs(self.state['angle']) > self.critical_angle

        self.signals['left'], self.signals['right'] = signals

        return feedback
