

class State(object):
    def __init__(self) -> None:
        super().__init__()
        self.done = False
        self.angle = 0.0
        self.angle_velocity = 0.0
        self.angle_acceleration = 0.0
        self.jerk = 0.0
        self.success = False

    def load(self, state):
        assert isinstance(state, State)
        self.done = state.done
        self.angle = state.angle
        self.angle_velocity = state.angle_velocity
        self.angle_acceleration = state.angle_acceleration
        self.jerk = state.jerk

class TwoMotorsStick(object):
    def __init__(self) -> None:
        super().__init__()
        self.stick_length = 1000. # all values in mm and gramms
        self.motor_center_ident = 50.
        self.regularator_center_ident = 100.
        self.motor_weight = 100. 
        self.regularator_weight = 50.
        self.stick_weight = 500.

        self.state = State()
        self.reward = 0
        self.done = False

    def update_state(self, forces):
        # assert isinstance(action, State)
        new_state = State().load(self.state)
        delta_force = forces[1] - forces[0]
        deltas = 0
        # REALIZE
        return deltas

    def get_reward(self, deltas):
        '''
        Get's reward according to the current state and its changings
        '''
        # REALIZE
        return 0

    def signal_to_force(self, signal):
        # REALIZE
        return signal * 10.

    def get_forces(self, action):
        return self.signal_to_force(action[0]), self.signal_to_force(action[1])

    def step(self, action):
        deltas = self.update_state(self.get_forces(action))
        self.reward += self.get_reward(deltas)

        if self.state.done:
            success_str = 'Success! ' if self.state.success else 'Fail! '
            print('Done! ' + success_str + 'Total reward per session: ', self.reward)
        return self.state.success, self.reward
        
        
class Session(object):
    def __init__(self) -> None:
        super().__init__()
        self.reset()
        
    def reset(self):
        self.model = TwoMotorsStick()
        self.done = self.model.done

    def run(self, n_iters):
        for i in range(n_iters):
            success, reward = self.model.step()
            if self.done:
                return success, reward
            


        

    