import torch
import torch.nn as nn

class TinyVGG(nn.Module):
    """Creates the TinyVGG architecture.
    Replicates the TinyVGG architecture from the CNN explainer website in PyTorch.
    See the original architecture here: https://poloclub.github.io/cnn-explainer/
    Args:
    in_channels: An integer indicating number of input channels.
    hidden_units: An integer indicating number of hidden units between layers.
    output_shape: An integer indicating number of output units.
    """

    def __init__(self, in_channels: int, hidden_units: int, output_shape: int):
        super().__init__()
        self.in_channels = in_channels
        self.output_shape = output_shape
        
        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=hidden_units, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units, out_channels=hidden_units, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.conv_block_2 = nn.Sequential(
            nn.Conv2d(in_channels=hidden_units, out_channels=hidden_units, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units, out_channels=hidden_units, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # Calculate the output size of convolutional blocks
        conv_output_size = self._get_conv_output_size(224)  # Input size is 224x224
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=conv_output_size, out_features=output_shape)
        )
    
    def _get_conv_output_size(self, input_size):
        # Helper method to calculate the output size of the convolutional blocks
        x = torch.randn(1, self.in_channels, input_size, input_size)
        x = self.conv_block_1(x)
        x = self.conv_block_2(x)
        return x.numel() // x.size(0)  # Total number of elements divided by batch size
    
    def forward(self, x):
        x = self.conv_block_1(x)
        x = self.conv_block_2(x)
        x = self.classifier(x)
        return x
import torch
import torch.nn as nn

class TinyVGG(nn.Module):
    def __init__(self, in_channels: int, hidden_units: int, output_shape: int):
        super().__init__()
        self.in_channels = in_channels
        self.output_shape = output_shape
        
        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=hidden_units, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units, out_channels=hidden_units, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.conv_block_2 = nn.Sequential(
            nn.Conv2d(in_channels=hidden_units, out_channels=hidden_units, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units, out_channels=hidden_units, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # Calculate the output size of convolutional blocks
        conv_output_size = self._get_conv_output_size(224)  # Input size is 224x224
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=conv_output_size, out_features=output_shape)
        )
    
    def _get_conv_output_size(self, input_size):
        # Helper method to calculate the output size of the convolutional blocks
        x = torch.randn(1, self.in_channels, input_size, input_size)
        x = self.conv_block_1(x)
        x = self.conv_block_2(x)
        return x.numel() // x.size(0)  # Total number of elements divided by batch size
    
    def forward(self, x):
        x = self.conv_block_1(x)
        x = self.conv_block_2(x)
        x = self.classifier(x)
        return x

