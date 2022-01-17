from utils import *
from utils import network_output_to_signal
from Copter.Network import Network
from model_parameters import MIN_SIGNAL, MAX_SIGNAL

import torch
import torch.nn.functional as F
from typing import List, Tuple, Dict, Type

class TwoMotorsStick(object):
    '''
    An environment for simulations. 
    A model of stick with 2 motors.
    If model_type is binary, every iteration each motor signal is increased or decreased on reaction_speed.
    '''
    def __init__(self, network, **kwargs) -> None:
        super().__init__()
        self.J = compute_total_J() # computes inertia moment of model
        self.critical_angle = get_max_angle() * 0.9 # with angle is treated as done
        assert network.in_channels == 3 # state tensor size
        assert network.out_channels == 4 
        self.network = network 
        self.model_type = 'binary' # continious type of model is depritiated

        self.reaction_speed = kwargs['reaction_speed'] 
        self.max_reward = kwargs['max_reward']
        self.angle_loss_coeff = kwargs['angle_loss_coeff'] 
        self.over_force_loss_coeff = kwargs['over_force_loss_coeff']
        self.upper_force_loss_coeff = kwargs['upper_force_loss_coeff']
        self.step_size = kwargs['step_size'] 
        self.target_upper_force = kwargs['target_upper_force'] 
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
            'left_signal':MIN_SIGNAL,
            'right_signal':MIN_SIGNAL,
        } 
        self.total_reward = 0 # total reward of environment
        self.done = False # if env is done (critial angle achieved)

    def predict_action_probs(self) -> Tuple[np.ndarray]:
        '''
        Predict distributions of actions according to the current state and using network
        Returns: list of 2 arrays - probability distributions for each motor
        '''
        with torch.no_grad():
            state_tensor = state_dict_to_tensor(self.state)
            state_tensor = add_noise(state_tensor)
            network_out = self.network(state_tensor)
            if self.model_type == 'continious':
                return network_out
            reshaped_network_out = torch.reshape(network_out, (2, -1))
            return F.softmax(reshaped_network_out[0], dim=-1).numpy(), F.softmax(reshaped_network_out[1], dim=-1).numpy()
            

    def compute_angle_acceleration(self, delta_force: float) -> float:
        '''
        Computes actual angle acceleration according to difference in forces
        '''
        return compute_acceleration_using_J(delta_force, self.J)

    def update_state(self, action: Tuple[int]) -> Dict[str, float]:
        '''
        Computes the differences of parameters and updates the current state
        Input: signals of motors (list of 2 tensors)
        Returns: difference between new and old states (dict)
        '''
        force_l = signal_to_force(self.state['left_signal'])
        force_r = signal_to_force(self.state['right_signal'])
        delta_force = force_r - force_l
        actual_angle_acceleration = self.compute_angle_acceleration(delta_force)
        deltas = {}
        new_angle = self.state['angle'] + self.state['angle_velocity'] * self.step_size
        new_angle_velocity = self.state['angle_velocity'] + self.state['angle_acceleration'] * self.step_size
        new_jerk = self.state['angle_acceleration'] - actual_angle_acceleration
        self.state['left_signal'] += self.reaction_speed * (action[0] * 2 - 1)
        self.state['right_signal'] += self.reaction_speed * (action[1] * 2 - 1)

        if abs(new_angle) > self.critical_angle:
            self.done = True
        
        deltas['angle'] = new_angle - self.state['angle']
        deltas['angle_velocity'] = new_angle_velocity - self.state['angle_velocity']
        deltas['angle_acceleration'] = actual_angle_acceleration - self.state['angle_acceleration']
        deltas['upper_force'] = math.cos(self.state['angle']) * (force_l + force_r)
        min_force, max_force = signal_to_force(MIN_SIGNAL), signal_to_force(MAX_SIGNAL)
        deltas['over_force'] = max(min_force - force_l, 0) + \
            max(min_force - force_l, 0) + \
                max(force_l - max_force, 0) + \
                    max(force_r - max_force, 0)
        
        self.state['angle'] = new_angle
        self.state['angle_velocity'] = new_angle_velocity
        self.state['angle_acceleration'] = actual_angle_acceleration
        self.state['angle_jerk'] = new_jerk
        return deltas
        
    def get_reward(self, deltas: Dict[str, float]) -> float:
        '''
        Get's reward according to the current state and its changings
        Input: difference between new and old states (dict)
        - angle diff
        - angle_velocity diff
        - angle_acceleration diff
        Returns: reward (float)
        '''
        losses = sum(self.get_losses_detailed(deltas))
        # assert self.max_reward > losses
        return max(self.max_reward - losses, 0)

    def get_losses_detailed(self, deltas: Dict[str, float]) -> List[float]:
        '''
        Gets components of losses of model
        '''
        loss_angle = self.angle_loss_coeff * abs(deltas['angle'])**2
        loss_upper_force = self.upper_force_loss_coeff * abs(deltas['upper_force'] - self.target_upper_force)**2
        return [loss_angle, loss_upper_force]

    def step(self) -> Tuple[float, Tuple[int], bool, None]:
        '''
        Updates state, making one step of model
        Input: an array of actions i.e. signals on motors
        Returns: 
        - reward (float)
        - action (array)
        - done (bool)
        - additioanal info (str)
        '''
        # if self.model_type == 'continious':
        #     network_output_means = self.predict_action_probs()
        #     means = network_output_to_signal(network_output_means)
        #     action = sample_actions(False, means[0], means[1], self.std)
        # else:
        signal_distr_l, signal_distr_r = self.predict_action_probs()
        action = sample_actions(signal_distr_l, signal_distr_r) # action is a tuple of 0 and 1
        state_difference = self.update_state(action)
        reward = self.get_reward(state_difference)
        self.total_reward += reward

        info = self.get_losses_detailed(state_difference) # for debug
        # info = None

        return reward, action, self.done, info
        