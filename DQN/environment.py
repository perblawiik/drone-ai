import gym
import torch
import numpy as np
import torchvision.transforms as T  

class CartPoleEnvManager():
    def __init__(self, device):
        """ Initializes the environment.
        Params
        ===
            device: The device that PyTorch to use for tensor calculations (CPU or GPU)
        """
        self.device = device
        self.env = gym.make('CartPole-v0').unwrapped
        self.env.reset()
        self.current_screen = None
        self.done = False

    def reset(self):
        self.env.reset()
        self.current_screen = None

    def close(self):
        self.env.close()

    def render(self, mode='human'):
        return self.env.render(mode)

    def num_actions_available(self):
        return self.env.action_space.n

    def take_action(self, action):
        """
        Params
            action(tensor): The action to take given as a tensor
            return(tensor): The reward wrapped in PyTorch tensor
        """
        # We are only interested in the reward and if the task is done
        # (Note that we’re calling item() on the action. This is because the action that will be passed to this function in our main program will be a tensor)
        _, reward, self.done, _ = self.env.step(action.item())
        # pylint: disable=not-callable
        return torch.tensor([reward], device=self.device)
        # pylint: enable=not-callable

    def in_starting_state(self):
        """ Returns True when the current_screen is None and returns 
            False otherwise.
        """
        return self.current_screen is None

    def get_state(self):
        """ Returns the current state of the environment in the form 
            of a processed image of the screen.
        """
        if self.in_starting_state() or self.done:
            self.current_screen = self.get_processed_screen()
            
            # Our starting state is represented by a black screen
            # pylint: disable=E1101
            black_screen = torch.zeros_like(self.current_screen)
            # pylint: enable=E1101
            return black_screen
        else:
            # A single state in the environment is represented as the difference between the current screen and the previous screen
            previous = self.current_screen
            current = self.get_processed_screen()

            # Update current screen
            self.current_screen = current
            return current - previous

    def get_screen_height(self):
        """ Computes the height of the processed screen by indexing 
            into the shape of the screen with a 2 to get the height.
        """
        screen = self.get_processed_screen()
        return screen.shape[2]

    def get_screen_width(self):
        """ Computes the width of the processed screen by indexing 
            into the shape of the screen with a 3 to get the width.
        """
        screen = self.get_processed_screen()
        return screen.shape[3]

    def get_processed_screen(self):
        # Render the environment for the current state into an RGB array and transpose it into the order (Channels, Height, Width)
        screen = self.render('rgb_array').transpose((2, 0, 1)) # PyTorch expects CHW
        screen = self.crop_screen(screen)
        return self.transform_screen_data(screen)

    def crop_screen(self, screen):
        """ Takes an image screen and returns a cropped version of it.
        Params
        ===
            screen: The image to crop
        """
        screen_height = screen.shape[1]

        # Remove empty space by stripping off top and bottom
        top = int(screen_height * 0.4) # Set top border to what corresponds to 40% of the full height
        bottom = int(screen_height * 0.8) # Set bottom border to what corresponds to 80% of the full height

        # Slice off 40% of the top and 20% of the bottom from the original screen
        screen = screen[:, top:bottom, :]
        return screen


    def transform_screen_data(self, screen):
        """ Takes an image screen and returns a transformed version of it.
        Params
        ===
            screen: The image to transform
        """
        # Convert the image to a contiguous array
        # (the values of will be stored sequentially next to each other in memory)
        screen = np.ascontiguousarray(screen, dtype=np.float32) / 255 # Rescale the pixel values into the normalized range [0, 1]

        # Convert screen array to a PyTorch tensor
        # pylint: disable=E1101
        screen = torch.from_numpy(screen)
        # pylint: enable=E1101

        # Use torchvision package to compose image transforms
        resize = T.Compose([
            T.ToPILImage() # Convert to a PIL image
            ,T.Resize((40,90)) # Resized to a 40 x 90 image
            ,T.ToTensor() # Transform to a tensor
        ])
        
        # Pass the screen to resize transform and then add a batch dimension to the tensor with unsqueeze()
        return resize(screen).unsqueeze(0).to(self.device) # Add a batch dimension (BCHW)