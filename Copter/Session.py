from TwoMotorStick import TwoMotorsStick

class Session(object):
    def __init__(self) -> None:
        super().__init__()
        self.reset()

        self.state_history = []
        self.action_history = []
        self.reward_history = []
        self.success = None
        self.done = None
        
    def reset(self):
        self.model = TwoMotorsStick()
        self.done = False

    def run(self, n_iters=100, reset=True):
        if reset:
            self.reset()
        for i in range(n_iters):
            self.state_history.append(self.model.state)
            reward, action, done, info = self.model.step()
            self.action_history.append(action)
            self.reward_history.append(reward)
            if self.done:
                self.success = False
                break
        if self.success is None:
            self.success = True