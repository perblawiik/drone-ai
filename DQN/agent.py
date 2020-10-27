import random
import torch

class DQNAgent():
    """ Reinforcement Learning Agent.
    """
    def __init__(self, strategy, num_actions, device):
        """
        Params
        ===
            strategy: Learning strategy
            num_actions(int): Number of possible actions the agent can take
            device: The device that PyTorch to use for tensor calculations (CPU or GPU)
        """
        # Counter for keeping track of the current time step
        self.current_step = 0
        self.strategy = strategy
        self.num_actions = num_actions
        self.device = device

    def select_action(self, state, policy_network):
        """ Selects which action to take.
        Params
        ===
            state: Current state
            policy_network: The Deep Q-Network for learning the optimal policy
            return (int): An index mapped to the action to take
        """
        # Compute the exploration rate based on the used strategy
        rate = self.strategy.get_exploration_rate(self.current_step)
        self.current_step += 1

        if rate > random.random():
            action = random.randrange(self.num_actions)
            # pylint: disable=not-callable
            return torch.tensor([action]).to(self.device) # explore 
            # pylint: disable=not-callable
        else:
            # turn off gradient tracking since we’re currently using the model for inference and not training
            with torch.no_grad():
                return policy_network(state).argmax(dim=1).to(self.device) # exploit