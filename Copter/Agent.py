# from utils import *
from model_parameters import MIN_SIGNAL, MAX_SIGNAL

import torch
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Dict, Type

class Agent(object):
    '''
    An environment for simulations. 
    A model of stick with 2 motors.
    If model_type is binary, every iteration each motor signal is increased or decreased on reaction_speed.
    '''
    def __init__(self, **kwargs) -> None:
        super().__init__()

        self.reaction_speed = kwargs['reaction_speed'] 
        self.max_reward = kwargs['max_reward']
        self.angle_loss_coeff = kwargs['angle_loss_coeff'] 
        self.over_force_loss_coeff = kwargs['over_force_loss_coeff']
        self.upper_force_loss_coeff = kwargs['upper_force_loss_coeff']
        self.step_size = kwargs['step_size'] 
        self.noise_std_out_signal = 0.0
        self.reset()
        # self.falied = False

    def reset(self):
        self.signals = {
            'left': MIN_SIGNAL + 100,
            'right': MIN_SIGNAL + 100,
        }
        self.losses = {
            'angle': None,
            'upper_force': None,
        }
        self.actions = {
            'left': None,
            'right': None,
        }
    
    def set_state(self, state: Dict[str, float]) -> None:
        self.state = state

    def set_target_upper_force(self, target_force: float) -> None:
        self.target_upper_force = target_force

    def get_signals(self) -> Tuple[int]:
        return self.signals['left'], self.signals['right']

    def make_signal_to_network(self) -> torch.Tensor:
        '''
        Transforms dict of state with 'angle', 'angle_velocity', 'angle_acceleration' keys to tensor of shape (3,)
        Then adds noize
        '''
        # assert isinstance(state_dict, dict)
        state_list = [
            self.state['angle'], 
            self.state['angle_velocity'], 
            self.state['angle_acceleration'],
            self.signals['left'],
            self.signals['right'],
            self.target_upper_force,
        ]
        state_tensor = torch.tensor(state_list, dtype=torch.float) 
        return state_tensor + torch.randn(state_tensor.shape) * self.noise_std_out_signal

    def sample_actions(self, action_probs_l: np.ndarray, action_probs_r: np.ndarray) -> Tuple[int]:
        '''
        Sample signals on motors.
        Input: list of np.arrays. Distribution parameters. 
        '''
        # if binary:
        a_l = np.random.choice(3, 1, p=action_probs_l)[0]
        a_r = np.random.choice(3, 1, p=action_probs_r)[0]
        return a_l, a_r

    def constrain_signals(self):
        self.signals['left'] = max(MIN_SIGNAL, self.signals['left'])
        self.signals['right'] = max(MIN_SIGNAL, self.signals['right'])
        self.signals['left'] = min(MAX_SIGNAL, self.signals['left'])
        self.signals['right'] = min(MAX_SIGNAL, self.signals['right'])

    def get_signal_from_network(self, network_out) -> torch.Tensor:
        reshaped_network_out = torch.reshape(network_out, (2, -1))
        signal_distribution_left = F.softmax(reshaped_network_out[0], dim=-1).numpy() 
        signal_distribution_rignt = F.softmax(reshaped_network_out[1], dim=-1).numpy()
        action = self.sample_actions(signal_distribution_left, signal_distribution_rignt) # action is a tuple of 0 and 1
        self.actions['left'] = action[0]
        self.actions['right'] = action[1]
        self.signals['left'] += self.reaction_speed * (action[0] - 1)
        self.signals['right'] += self.reaction_speed * (action[1] - 1)
        self.constrain_signals()
        
    def get_reward(self) -> float:
        '''
        Get's reward according to the current state and its changings
        Returns: reward (float)
        '''
        losses = sum(self.losses.values())
        return max(self.max_reward - losses, 0)

    def get_losses(self, feedback: Dict[str, float]) -> None:
        '''
        Computes components of losses of model
        Input: difference between new and old states (dict)
        - angle diff
        - angle_velocity diff
        - angle_acceleration diff
        '''
        self.losses['angle'] = self.angle_loss_coeff * abs(feedback['delta_angle'] + self.state['angle'])
        self.losses['upper_force'] = self.upper_force_loss_coeff * abs(feedback['upper_force'] - self.target_upper_force)
    
    def is_failed(self, feedback: Dict[str, float]) -> bool:
        return feedback['failed'] 
        