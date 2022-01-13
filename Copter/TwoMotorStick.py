from utils import *
from utils import network_output_to_signal
from Copter.Network import Network
from model_parameters import MIN_SIGNAL, MAX_SIGNAL

import torch
import torch.nn.functional as F

class TwoMotorsStick(object):
    def __init__(self, network, model_type='binary', target_upper_force=0.0, jerk_loss_coeff=1.0, step_size=1e-1, std=0.01) -> None:
        super().__init__()
        self.J = compute_total_J()
        self.critical_angle = get_max_angle() * 0.9

        assert model_type in ['binary', 'continious']
        assert network.in_channels == 3
        if model_type == 'continious':
            assert network.out_channels == 2
        else:
            assert network.out_channels == 4
        self.model_type = model_type
        self.network = network # Network(3, 2)
        self.reaction_speed = 10
        self.jerk_loss_coeff = jerk_loss_coeff 
        self.upper_force_loss_coeff = 1
        self.step_size = step_size # size of step in seconds
        self.std = std # variance of distribution
        self.target_upper_force = target_upper_force

        self.state = {
            'angle':torch.tensor(0, dtype=torch.float),
            'angle_velocity':torch.tensor(0, dtype=torch.float),
            'angle_acceleration':torch.tensor(0, dtype=torch.float),
            'angle_jerk':torch.tensor(0, dtype=torch.float),
            'left_signal':torch.tensor(MIN_SIGNAL, dtype=torch.float),
            'right_signal':torch.tensor(MIN_SIGNAL, dtype=torch.float),
        }
        self.total_reward = 0
        self.done = False
        self.success = False

    def predict_action_probs(self):
        '''
        Predict probabilities of actions according to the current state and using model
        Input: state (dict)
        Returns: list of 2 tensors - probability distributions
        '''
        with torch.no_grad():
            state_tensor = state_dict_to_tensor(self.state)
            state_tensor = add_noise(state_tensor)
            network_out = self.network(state_tensor)
            if self.model_type == 'continious':
                return network_out
            reshaped_network_out = torch.reshape(network_out, (2, -1))
            return F.softmax(reshaped_network_out[0], dim=-1), F.softmax(reshaped_network_out[1], dim=-1)
            

    def compute_angle_acceleration(self, delta_force):
        '''
        Computes actual angle acceleration according to difference in forces
        '''
        return compute_acceleration_using_J(delta_force, self.J)

    def update_state(self, action_l, action_r):
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
        self.state['left_signal'] += self.reaction_speed * action_l
        self.state['right_signal'] += self.reaction_speed * action_r

        if abs(new_angle) > self.critical_angle:
            self.done = True
        
        deltas['angle'] = new_angle - self.state['angle']
        deltas['angle_velocity'] = new_angle_velocity - self.state['angle_velocity']
        deltas['angle_acceleration'] = actual_angle_acceleration - self.state['angle_acceleration']
        deltas['upper_force'] = torch.cos(self.state['angle']) * (force_l + force_r)
        min_force, max_force = torch.tensor(signal_to_force(MIN_SIGNAL)), torch.tensor(signal_to_force(MAX_SIGNAL))
        deltas['over_force'] = F.relu(min_force - force_l) + \
            F.relu(min_force - force_l) + \
                F.relu(force_l - max_force) + \
                    F.relu(force_r - max_force)
        
        self.state['angle'] = new_angle
        self.state['angle_velocity'] = new_angle_velocity
        self.state['angle_acceleration'] = actual_angle_acceleration
        self.state['angle_jerk'] = new_jerk
        return deltas
        

    def get_reward(self, deltas):
        '''
        Get's reward according to the current state and its changings
        Input: difference between new and old states (dict)
        - angle diff
        - angle_velocity diff
        - angle_acceleration diff
        Returns: reward (float)
        '''
        loss = torch.square(deltas['angle']) + \
            self.jerk_loss_coeff * torch.square(deltas['angle_acceleration']) + \
                self.upper_force_loss_coeff * torch.square(deltas['upper_force'] - self.target_upper_force) +\
                    deltas['over_force'] * 1000.
        # proportion = (torch.square(deltas['angle']) / torch.square(deltas['upper_force'] - self.target_upper_force)).item()
        # print(proportion)
        return -loss

    def get_force(self, action):
        '''
        Computes the difference between right and left motor forces according to input signals
        Input: list of 2 signals (list of torch.tesnors)
        Returns: list of 2 forces (list of torch.tesnors)
        '''
        return signal_to_force(action[0]), signal_to_force(action[1])

    def step(self):
        '''
        Updates state, making one step of model
        Input: an array of actions i.e. signals on motors
        Returns: 
        - reward (float)
        - action (array)
        - done (bool)
        - additioanal info (str)
        '''
        if self.model_type == 'continious':
            network_output_means = self.predict_action_probs()
            means = network_output_to_signal(network_output_means)
            action = sample_actions(False, means[0], means[1], self.std)
        else:
            signal_distr_l, signal_distr_r = self.predict_action_probs()
            action = sample_actions(True, signal_distr_l, signal_distr_r)
        state_difference = self.update_state(action[0], action[1])
        reward = self.get_reward(state_difference)
        self.total_reward += reward.item()

        return reward, action, self.done, None
        