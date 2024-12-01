import numpy as np
import torch
import torch.nn as nn

class MyCNN(nn.Module):

  def __init__(self, input_channels: int = 1, num_classes: int = 20, activation_function: torch.nn.Module = torch.nn.ReLU()):
    super().__init__()
    self.activation = activation_function
    self.conv0 = torch.nn.Conv2d(in_channels=input_channels, out_channels=32, kernel_size=3) # (1,100,100) to (32,98,98)
    self.conv1 = torch.nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3) # (32,98,98) to (32,96,96)
    self.conv2 = torch.nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3) # (32,96,96) to (32,94,94)
    self.maxpool0 = torch.nn.MaxPool2d(kernel_size=2) # (32,94,94) to (32,47,47)
    self.conv3 = torch.nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3) # (32,47,47) to (32,45,45)
    self.conv4 = torch.nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3) # (32,45,45) to (32,43,43)
    self.conv5 = torch.nn.Conv2d(in_channels=32, out_channels=32, kernel_size=5) # (32,43,43) to (32,39,39)
    self.maxpool1 = torch.nn.MaxPool2d(kernel_size=3) # (32,39,39) to (32,13,13)
    self.conv6 = torch.nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3) # (32,13,13) to (32,11,11)
    self.conv7 = torch.nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3) # (32,11,11) to (32,9,9)
    self.maxpool2 = torch.nn.MaxPool2d(kernel_size=3) # (32,9,9) to (32,3,3)
    self.flatten = torch.nn.Flatten() # (32,3,3) to 1D shape of size (32,3,3)
    self.linear0 = torch.nn.Linear(32*3*3, 128)
    self.linear1 = torch.nn.Linear(128, 64)
    self.linear2 = torch.nn.Linear(64, num_classes)
    
  def forward(self, input_images: torch.Tensor):
    x = input_images
    x = self.conv0(x)
    x = self.activation(x)
    x = self.conv1(x)
    x = self.activation(x)
    x = self.conv2(x)
    x = self.activation(x)
    x = self.maxpool0(x)
    x = self.conv3(x)
    x = self.activation(x)
    x = self.conv4(x)
    x = self.activation(x)
    x = self.conv5(x)
    x = self.activation(x)
    x = self.maxpool1(x)
    x = self.conv6(x)
    x = self.activation(x)
    x = self.conv7(x)
    x = self.activation(x)
    x = self.maxpool2(x)
    x = self.flatten(x)
    x = self.linear0(x)
    x = self.activation(x)
    x = self.linear1(x)
    x = self.activation(x)
    x = self.linear2(x)
    return x
    
model =  MyCNN()