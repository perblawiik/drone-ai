import torch
import torch.nn as nn
import torch.nn.functional as F

class DeepQNetwork(nn.Module):
    """Policy Q-Network Model. """
    def __init__(self, img_height, img_width, num_actions=2):

        """Initialize parameters and build model.
           The Deep Q-Network will receive screenshot images as input, so heigh and width are required.
        Params
        ======
            img_height (int): The pixel height of the images
            img_width (int): The pixel width of the images
            num_actions (int): Number of available actions
        """
        super().__init__()

        # The network consist of two fully connected hidden layers (fc1 & fc2), and one output layer (out)
        # (PyTorch refers to fully connected layers as Linear layers)

        # Number of inputs in the first layer corresponds to the number of pixels of the input image 
        # (width * height * num_color_channels)
        self.fc1 = nn.Linear(in_features=img_height*img_width*3, out_features=24)   
        self.fc2 = nn.Linear(in_features=24, out_features=32)
        self.out = nn.Linear(in_features=32, out_features=num_actions)

    def forward(self, t):
        """ Forward pass through the network. 
            (Note that all PyTorch neural networks require an implementation of forward())
        """
        # Any image tensor, t, passed to the network, will first need to be flattened
        t = t.flatten(start_dim=1)
        # Pass flattened image into first layer
        t = F.relu(self.fc1(t))
        # Pass output from first layer into second layer
        t = F.relu(self.fc2(t))
        # Pass output from second layer into output layer
        t = self.out(t)

        # Return actions
        return t

# pylint: disable=E1101
class QValues():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @staticmethod
    def get_current(policy_network, states, actions):
        """ Returns the predicted Q-Values from the given policy network based on the specified states and actions
        """
        return policy_network(states).gather(dim=1, index=actions.unsqueeze(-1))


    @staticmethod        
    def get_next(target_network, next_states):
        """ Returns the predicted maximum q-values from the target network for specified next states
        """
        # Find the locations of all the final states (final states are defined by black screens)
        final_state_locations = next_states.flatten(start_dim=1).max(dim=1)[0].eq(0).type(torch.bool)

        # Filter out final states locations
        non_final_state_locations = (final_state_locations == False)
        non_final_states = next_states[non_final_state_locations]

        values = torch.zeros(next_states.shape[0]).to(QValues.device)
        values[non_final_state_locations] = target_network(non_final_states).max(dim=1)[0].detach()
        return values

# pylint: enable=E1101