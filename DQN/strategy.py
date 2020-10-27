import math

class EpsilonGreedyStrategy():
    def __init__(self, start, end, decay):
        """ Initialize epsilon parameters
            start(float): Initial epsilon value
            end(float): Minimum epsilon value
            decay(float): Epsilon decay rate
        """
        self.start = start
        self.end = end
        self.decay = decay

    def get_exploration_rate(self, current_step):
        """ Determines the exploration rate based on exponential decay.
        Params
        ===
            current_step(int): Current time step of the agent
            return: The calculated exploration rate
        """
        return self.end + (self.start - self.end) * math.exp(-1. * current_step * self.decay)