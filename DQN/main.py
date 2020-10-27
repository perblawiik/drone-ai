import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from itertools import count
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# Custom classes
from environment import CartPoleEnvManager
from strategy import EpsilonGreedyStrategy
from agent import DQNAgent
from replay_memory import ReplayMemory
from replay_memory import Experience
from dqn_model import DeepQNetwork
from dqn_model import QValues

is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython: from IPython import display

#----- UTILITY FUNCTIONS -----#
def plot(values, moving_avg_period):
    plt.figure(2)
    plt.clf()        
    plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(values)

    moving_avg = get_moving_average(moving_avg_period, values)
    plt.plot(moving_avg)    
    plt.pause(0.001)
    print("Episode", len(values), "\n", \
        moving_avg_period, "episode moving avg:", moving_avg[-1])
    if is_ipython: display.clear_output(wait=True)

    return moving_avg[-1]

# pylint: disable=E1101
# pylint: disable=not-callable

def get_moving_average(period, values):
    values = torch.tensor(values, dtype=torch.float)

    # Make sure we have enough values
    if len(values) >= period:
        moving_avg = values.unfold(dimension=0, size=period, step=1) \
            .mean(dim=1).flatten(start_dim=0)
        moving_avg = torch.cat((torch.zeros(period-1), moving_avg))
        return moving_avg.numpy()
    else:
        moving_avg = torch.zeros(len(values))
        return moving_avg.numpy()


def extract_tensors(experiences):
    # Convert batch of Experiences to Experience of batches
    batch = Experience(*zip(*experiences))

    # Create separate lists for the different attributes
    t1 = torch.cat(batch.state)
    t2 = torch.cat(batch.action)
    t3 = torch.cat(batch.reward)
    t4 = torch.cat(batch.next_state)

    return (t1,t2,t3,t4)

# pylint: enable=E1101
# pylint: enable=not-callable

#----- Hyperparameters -----#
#---------------------------#

batch_size = 128
gamma = 0.999 # Discount factor used in the Bellman equation
epsilon_start = 1.0
epsilon_end = 0.005
epsilon_decay = 0.0005
target_update_rate = 10 # How often the target network weights should be updated with policy network weights
memory_capacity = 100000 # Replay memory size
learning_rate = 0.001
num_episodes = 1600

#----- SETUP PHASE -----#
#-----------------------#

# pylint: disable=E1101
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# pylint: enable=E1101

environment = CartPoleEnvManager(device)
strategy = EpsilonGreedyStrategy(epsilon_start, epsilon_end, epsilon_decay)

agent = DQNAgent(strategy, environment.num_actions_available(), device)
memory = ReplayMemory(memory_capacity)

# Create DQN networks and put them on the defined device
policy_network = DeepQNetwork(environment.get_screen_height(), environment.get_screen_width()).to(device)
target_network = DeepQNetwork(environment.get_screen_height(), environment.get_screen_width()).to(device)

# Set the weights and biases in the target network to be the same as in the policy network
target_network.load_state_dict(policy_network.state_dict())

# Tell PyTorch that this network is not in training mode. This network will only be used for inference
target_network.eval()

# Set optimizer to the Adam algorithm
optimizer = optim.Adam(params=policy_network.parameters(), lr=learning_rate)

#----- TRAINING PHASE -----#
#--------------------------#

# A list for storing episode durations used for plotting
episode_durations = []
average_score = 0

for episode in range(num_episodes):
    # Reset environment to begin new episode
    environment.reset()

    # Get initial state
    current_state = environment.get_state() 

    for timestep in count():
        # Select action based on current state, retrieve reward and get next state
        action = agent.select_action(current_state, policy_network)
        reward = environment.take_action(action)
        next_state = environment.get_state()

        # Add experience to the replay memory
        memory.push(Experience(current_state, action, next_state, reward))

        # Update current state
        current_state = next_state

        # Make sure the memory has enough experiences to get a sample batch
        if memory.can_provide_sample(batch_size):
            # Retrieve a sample batch from replay memory to train the policy network
            experiences = memory.sample(batch_size)
            states, actions, rewards, next_states = extract_tensors(experiences)

            current_q_values = QValues.get_current(policy_network, states, actions)
            next_q_values = QValues.get_next(target_network, next_states)
            target_q_values = (next_q_values * gamma) + rewards

            # Use mean squared error to compute the loss between the current_q_values and the target_q_values
            loss = F.mse_loss(current_q_values, target_q_values.unsqueeze(1))

            # Clear the gradients
            optimizer.zero_grad()

            # Compute the gradient of the loss based on the weights and biases in policy_network
            loss.backward()
            optimizer.step() # Update the weights and biases with computed gradients

            # End loop if the task is complete
            if environment.done:
                episode_durations.append(timestep)
                average_score = plot(episode_durations, 100)
                break
    
    # Use the policy network to update the target network
    if episode % target_update_rate == 0:
        target_network.load_state_dict(policy_network.state_dict())

torch.save(policy_network.state_dict(), 'checkpoint_' + str(num_episodes) + '_episodes_' + str(round(average_score)) + '_score.pth')

# policy_network.load_state_dict(torch.load('dqn_1000_episodes.pth'))
# for i in range(10):
#     # Reset environment to begin new episode
#     environment.reset()

#     # Get initial state
#     current_state = environment.get_state() 

#     for timestep in count():
#         # Select action based on current state, retrieve reward and get next state
#         action = agent.select_action(current_state, policy_network)
#         reward = environment.take_action(action)
#         environment.render()

#         if environment.done:
#             break 


environment.close()