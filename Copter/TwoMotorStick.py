from utils import *
from utils import network_output_to_signal
from Copter.Network import Network

import torch
import torch.nn.functional as F

class TwoMotorsStick(object):
    def __init__(self, network, model_type='continious', target_upper_force=12, jerk_loss_coeff=1.0, step_size=1e-1, std=0.01) -> None:
        super().__init__()
        self.J = compute_total_J()
        self.critical_angle = get_max_angle() * 0.9

        assert model_type in ['binary', 'continious']
        self.model_type = model_type
        self.network = network # Network(3, 2)
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
        }
        self.total_reward = 0
        self.done = False
        self.success = False

    def predict_action_probs(self):
        '''
        Predict probabilities of actions according to the current state and using model
        '''
        with torch.no_grad():
            state_tensor = state_dict_to_tensor(self.state)
            state_tensor = add_noise(state_tensor)
            network_out = self.network(state_tensor)
            if self.model_type == 'continious':
                return network_out
            return F.softmax(network_out, dim=-1)
            

    def compute_angle_acceleration(self, delta_force):
        '''
        Computes actual angle acceleration according to difference in forces
        '''
        return compute_acceleration_using_J(delta_force, self.J)

    def update_state(self, force_l, force_r):
        '''
        Computes the differences of parameters and updates the current state
        Input: forces of motors (float)
        Returns: difference between new and old states (dict)
        '''
        delta_force = force_r - force_l
        actual_angle_acceleration = self.compute_angle_acceleration(delta_force)
        deltas = {}
        new_angle = self.state['angle'] + self.state['angle_velocity'] * self.step_size
        new_angle_velocity = self.state['angle_velocity'] + self.state['angle_acceleration'] * self.step_size
        new_jerk = self.state['angle_acceleration'] - actual_angle_acceleration

        if abs(new_angle) > self.critical_angle:
            self.done = True
        
        deltas['angle'] = new_angle - self.state['angle']
        deltas['angle_velocity'] = new_angle_velocity - self.state['angle_velocity']
        deltas['angle_acceleration'] = actual_angle_acceleration - self.state['angle_acceleration']
        deltas['upper_force'] = torch.cos(self.state['angle']) * (force_l + force_r)
        
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
                self.upper_force_loss_coeff * torch.square(deltas['upper_force'] - self.target_upper_force)
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
        network_output_means = self.predict_action_probs()
        means = network_output_to_signal(network_output_means)
        action = sample_actions(means[0], means[1], self.std)
        force_l, force_r = self.get_force(action)
        state_difference = self.update_state(force_l, force_r)
        reward = self.get_reward(state_difference)
        self.total_reward += reward.item()

        return reward, action, means, self.done, None
        