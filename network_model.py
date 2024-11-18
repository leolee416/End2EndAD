
"""
  *  @copyright (c) 2020 Charan Karthikeyan P V, Nagireddi Jagadesh Nischal
  *  @file    network_model.py
  *  @author  Charan Karthikeyan P V, Nagireddi Jagadesh Nischal
  *
  *  @brief Class declaration for building the network model.  
 """

import torch
import torchvision.models as models
import torch.nn as nn
from torchsummary import summary
"""
* @brief Class declaration for building the training network model.
* @param None.
* @return The built netwwork.
"""
### 时序结构CNN模型
class model_cnn(nn.Module):
    def __init__(self):
        super().__init__()

        self.elu = nn.ELU()    #激活函数ELU
        Resnet50 = models.resnet50(pretrained = True)   #加载预训练Resnet50
        self.model = torch.nn.Sequential(*list(Resnet50.children())[:-3])  #把最后的layer4，avgpool和FC去除
        self.drop_1 = nn.Dropout(0.5) #dropout多余参数
        self.Flatten = nn.Flatten()   #归一化展平
        self.fc1 = nn.Linear(200704,100)  #第一个FC
        self.drop_2 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(100,50)  #第二个FC
        self.drop_3 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(50,1) #第三个FC输出归一后的转向角
        # self.drop_4 = nn.Dropout(0.5)
        # self.fc4 = nn.Linear(100,1)
        
    def forward(self, input):
        input = input/255
        input = self.model(input)
        input = self.drop_1(input)
        input = self.Flatten(input)
        input = self.elu(self.fc1(input))
        input = self.drop_2(input)
        input = self.elu(self.fc2(input))
        input = self.drop_3(input)
        input = self.elu(self.fc3(input))
        # input = self.drop_4(input)
        # input = self.fc4(input)

        return input
    