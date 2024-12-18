import torch
from torch import nn
import torchvision
from torchvision import transforms

class CNN_Model(nn.Module):
  def __init__(self,input_layer,hidden_layer,output_layer):
    super().__init__()
    self.block_1 = nn.Sequential(
        nn.Conv2d(in_channels=input_layer,
                  out_channels = hidden_layer,
                  kernel_size=3,
                  stride=1,
                  padding=1,
                  ),
        nn.ReLU(),
        nn.Conv2d(in_channels = hidden_layer,
                  out_channels = hidden_layer,
                  kernel_size=3,
                  stride=1,
                  padding=1,
                  ),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2,
                     stride=2,
                     )
    )

    self.block_2 = nn.Sequential(
        nn.Conv2d(in_channels=hidden_layer,
                  out_channels = hidden_layer,
                  kernel_size=3,
                  stride=1,
                  padding=1,
                  ),
        nn.ReLU(),
        nn.Conv2d(in_channels = hidden_layer,
                  out_channels = hidden_layer,
                  kernel_size=3,
                  stride=1,
                  padding=1,
                  ),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2,
                     stride=2,
                     )
    )

    self.classifier = nn.Sequential(
        nn.Flatten(),
        nn.Linear(in_features = hidden_layer * 7 *7,
                  out_features = output_layer,
                  )
    )
  def forward(self,x):
    return self.classifier(self.block_2(self.block_1(x)))
