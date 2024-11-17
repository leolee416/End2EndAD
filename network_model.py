
"""
  *  @copyright (c) 2020 Charan Karthikeyan P V, Nagireddi Jagadesh Nischal
  *  @file    network_model.py
  *  @author  Charan Karthikeyan P V, Nagireddi Jagadesh Nischal
  *
  *  @brief Class declaration for building the network model.  
 """

import torch
import torch.nn as nn
from torchsummary import summary
from torchvision import models
"""
* @brief Class declaration for building the training network model.
* @param None.
* @return The built netwwork.
"""
class model_cnn(nn.Module):
    """
    * @brief Initializes the class varaibles
    * @param None.
    * @return None.
    """
    def __init__(self):
        super().__init__()

        self.elu = nn.ELU()
        self.dropout = nn.Dropout()

        self.conv_0 = nn.Conv2d(3, 24, 5, stride=2)
        self.conv_1 = nn.Conv2d(24, 36, kernel_size=5, stride=2)
        self.conv_2 = nn.Conv2d(36, 48, kernel_size=5, stride=2) #384 kernels, size 3x3
        self.conv_3 = nn.Conv2d(48, 64, kernel_size=3) # 384 kernels size 3x3
        self.conv_4 = nn.Conv2d(64, 64, kernel_size=3) # 256 kernels, size 3x3

        self.fc0 = nn.Linear(1152, 100)
        self.fc1 = nn.Linear(100,50)
        self.fc2 = nn.Linear(50, 10)
        self.fc3 = nn.Linear(10, 1)
    """ 
    * @brief Function to build the model.
    * @parma The image to train.
    * @return The trained prediction network.
    """
    def forward(self, input):
        input = input/127.5-1.0
        input = self.elu(self.conv_0(input))
        input = self.elu(self.conv_1(input))
        input = self.elu(self.conv_2(input))
        input = self.elu(self.conv_3(input))
        input = self.elu(self.conv_4(input))
        input = self.dropout(input)

        input = input.flatten()
        input = self.elu(self.fc0(input))
        input = self.elu(self.fc1(input))
        input = self.elu(self.fc2(input))
        input = self.fc3(input)

        return input

def resnet_model(weights='ResNet50_Weights.DEFAULT'):
    model = models.resnet50(weights = weights)
    model.fc = nn.Linear(in_features=2048, out_features=1)
    return model

import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self, inChannels, outChannels, matchDim = None, stride = 1):
        super().__init__()
        self.matchDim = matchDim
        self.conv0 = nn.Sequential( 
                nn.Conv2d(
                    in_channels=inChannels,
                    out_channels=outChannels,
                    kernel_size=1,
                    stride=1
                ),
                nn.BatchNorm2d(outChannels),
                nn.ELU(),
        )

        self.conv1 = nn.Sequential( 
                nn.Conv2d(
                    in_channels=outChannels,
                    out_channels=outChannels,
                    kernel_size=3,
                    stride=stride,
                    padding=1
                ),
                nn.BatchNorm2d(outChannels),
                nn.ELU(),
        )

        self.conv2 = nn.Sequential( 
                nn.Conv2d(
                    in_channels=outChannels,
                    out_channels=outChannels*4,
                    kernel_size=1,
                    stride=1
                ),
                nn.BatchNorm2d(outChannels * 4),
        )

        self.elu = nn.Sequential(nn.ELU(),nn.Dropout(0.2),)

    
    def forward(self, x):
        residual = x
        out = self.conv0(x)
        out = self.conv1(out)
        out = self.conv2(out)
        
        if self.matchDim:
            residual = self.matchDim(residual)
        out += residual
        out = self.elu(out)

        return out


class Model(nn.Module):
    def __init__(self):
        super().__init__()


        self.res = nn.Sequential(
            #Conv1
            nn.Conv2d(
                in_channels=3,
                out_channels=64,
                kernel_size=5,
                stride = 2,
            ),
            nn.BatchNorm2d(64),
            nn.ELU(),

            #Conv2
            nn.MaxPool2d(
                kernel_size=3,
                stride = 2,
                padding = 1,
            ),

            self.getBlock(64, 64, 2),
            #Conv3
            self.getBlock(256, 128, 4, 2),
            #Conv4
            self.getBlock(512, 256, 6, 2),

            nn.Conv2d(
                in_channels=1024,
                out_channels=512,
                kernel_size=3,
            ),
            nn.ELU(),

            nn.Dropout(),
            
            nn.AdaptiveAvgPool2d((1, 1)),
            
        )

        self.lin = nn.Sequential(
            nn.Linear(in_features=512, out_features=50),
            nn.ELU(),
            nn.Linear(in_features=50, out_features=10),
            nn.ELU(),
            nn.Linear(in_features=10, out_features=1),
        )




    def getBlock(self, inChannels, outChannels, numLayers, stride=1):
        layers = []
        #This is used to match the dimenionality of the residual to the output of a residual block
        matchDim = nn.Sequential(
                nn.Conv2d(
                    inChannels,
                    outChannels*4,
                    kernel_size=1,
                    stride=stride
                ),
                nn.BatchNorm2d(outChannels*4),
            )
        
        layers.append(
            ResidualBlock(
                inChannels,
                outChannels,
                matchDim,
                stride,
            )
        )
        
        inChannels = outChannels*4
        for _ in range(numLayers-1):
            layers.append(
                ResidualBlock(
                    inChannels,
                    outChannels
                )
            )
        
        return nn.Sequential(*layers)

    def forward(self, input):
        input = input/127.5-1.0
        out = self.res(input).flatten()
        out = self.lin(out)

        return out
    