from utils import *
from Copter.Network import Network

import torch
import torch.nn.functional as F

class TwoMotorsStick(object):
    def __init__(self, jerk_loss_coeff=1.0, step_size=1e-1, std=100) -> None:
        super().__init__()
        self.J = compute_total_J()
        self.critical_angle = get_max_angle() * 0.9

        self.network = Network(3, 2)
        self.jerk_loss_coeff = jerk_loss_coeff 
        self.step_size = step_size # size of step in seconds
        self.std = std # variance of distribution

        self.state = {
            'angle':0.,
            'angle_velocity':0.,
            'angle_acceleration':0.,
            'angle_jerk':0.,
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
            probas = self.network(state_tensor)
            # probas = F.softmax(logits).numpy()
            return probas

    def compute_angle_acceleration(self, delta_force):
        '''
        Computes actual angle acceleration according to difference in forces
        '''
        return compute_acceleration_using_J(delta_force, self.J)

    def update_state(self, delta_force):
        '''
        Computes the differences of parameters and updates the current state
        Input: difference of forces of motors (float), size of step in seconds (float)
        Returns: difference between new and old states (dict)
        '''
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
        
        self.state['angle'] = new_angle
        self.state['velocity'] = new_angle_velocity
        self.state['accelaration'] = actual_angle_acceleration
        self.state['jerk'] = new_jerk
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
        loss = deltas['angle']**2 + self.jerk_loss_coeff * deltas['angle_acceleration']**2
        return -loss

    def get_delta_force(self, action):
        '''
        Computes the difference between right and left motor forces according to input signals
        Input: list of 2 signals (list)
        Returns: difference of forces (float)
        '''
        return signal_to_force(action[1]) - signal_to_force(action[0])

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
        means = self.predict_action_probs()
        action = sample_actions(means[0], means[1], self.std)
        delta_force = self.get_delta_force(action)
        state_difference = self.update_state(delta_force=delta_force)
        reward = self.get_reward(state_difference)
        self.total_reward += reward

        return reward, action, self.done, None
        