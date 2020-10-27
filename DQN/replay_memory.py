import random
from collections import namedtuple

# The function namedtuple() is returning a new tuple subclass named Experience
# that includes the fields ('state', 'action', 'next_state', 'reward')
Experience = namedtuple(
    'Experience',
    ('state', 'action', 'next_state', 'reward')
)

# Example of usin the Experience tuple
#e = Experience(state=2, action=3, next_state=1, reward=4) 

class ReplayMemory():
    """ Memory buffer for storing experiences tuples.
    """

    def __init__(self, capacity):
        """ Initializes the ReplayMemory object
        Params
        ======
            capacity(int): Memory capacity
            memory([]): Container that holds the stored experiences
            push_count(int): A counter for keeping track of the number of experiences added to memory.
        """
        self.capacity = capacity
        self.memory = []
        self.push_count = 0

    def push(self, experience):
        """ Stores experiences in replay memory.
        """
        # Check if the number of experiences stored in memory is less than the memory capacity
        if len(self.memory) < self.capacity:
            self.memory.append(experience)
        # Push new experiences into the front of the memory container (overwrites the oldest experiences)
        else: 
            self.memory[self.push_count % self.capacity] = experience
        
        # Increment memory counter
        self.push_count += 1

    def sample(self, batch_size):
        """  Returns a batch of random samples from collected experiences
        Params
        ======
            batch_size(int): The number of samples to return
        """
        return random.sample(self.memory, batch_size)


    def can_provide_sample(self, batch_size):
        """ Returns true if the batch_size does not exceed number of experiences stored in memory container
        """
        return len(self.memory) >= batch_size


