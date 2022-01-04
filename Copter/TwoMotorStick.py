# from State import State
from utils import *

class TwoMotorsStick(object):
    def __init__(self, net) -> None:
        super().__init__()
        self.state = {
            'done':False,
            'angle':0.,
            'angle_velocity':0.,
            'angle_acceleration':0.,
            'angle_jerk':0.,
        }
        self.network = net
        self.total_reward = 0
        self.done = False
        self.success = False

    def predict_action_probs(self):
        '''
        Predict probabilities of actions according to the current state and using model
        '''
        # REALIZE
        pass
        # return np.zeros(2)

    def update_state(self, delta_force):
        '''
        Computes the differences of parameters and updates the current state
        Input: difference of forces of motors (float)
        Returns: difference between new and old states (dict)
        '''
        # REALIZE
        pass

    def get_reward(self, deltas):
        '''
        Get's reward according to the current state and its changings
        Input: difference between new and old states (dict)
        Returns: reward (float)
        '''
        pass
        # REALIZE

    def get_delta_force(self, action):
        '''
        Computes the difference between right and left motor forces according to sinput signals
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
        self.state_history.append(self.state.copy())
        action = sample_actions(self.predict_action_probs())
        delta_force = self.get_delta_force(action)
        state_difference = self.update_state(delta_force=delta_force)
        reward = self.get_reward(state_difference)
        self.total_reward += reward

        # if self.state['done']:
            # success_str = 'Success! ' if self.success else 'Fail! '
            # info_str = 'Done! ' + success_str + 'Total reward per session: ' + str(self.total_reward)
        return reward, action, self.state['done'], None
        