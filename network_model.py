import torch
import torchvision.models as models
import torch.nn as nn

class ModelCNN(nn.Module):
    def __init__(self, output_fields=None):
        """
        初始化模型
        :param output_fields: 要预测的输出字段列表，如 ["Steering"] 或 ["Steering", "Throttle", "Brake"]
        """
        super().__init__()

        self.elu = nn.ELU()  # 激活函数 ELU
        Resnet50 = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)  # 加载预训练 ResNet50
        self.model = torch.nn.Sequential(*list(Resnet50.children())[:-3])  # 移除 layer4, avgpool 和 fc
        self.drop_1 = nn.Dropout(0.5)  # Dropout 层
        self.Flatten = nn.Flatten()  # 展平层

        # 输出字段，用于动态确定最后一层的输出
        self.output_fields = output_fields if output_fields else ["Steering"]
        self.num_outputs = len(self.output_fields)  # 输出维度

        # 全连接层在 forward 中动态初始化
        self.fc1 = None  
        self.drop_2 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(100, 50)  # 第二个全连接层
        self.drop_3 = nn.Dropout(0.5)
        self.fc3 = None  # 最后一层全连接层，动态初始化

    def forward(self, input):
        # 通过截断的 ResNet50
        input = self.model(input)
        input = self.drop_1(input)
        input = self.Flatten(input)  # 展平

        # 动态初始化 fc1 和 fc3
        if self.fc1 is None:
            in_features = input.size(1)  # 展平后的特征数
            self.fc1 = nn.Linear(in_features, 100).to(input.device)
        
        if self.fc3 is None:
            self.fc3 = nn.Linear(50, self.num_outputs).to(input.device)  # 输出对应多个字段

        # 通过全连接层
        input = self.elu(self.fc1(input))  # 第一个全连接层
        input = self.drop_2(input)
        input = self.elu(self.fc2(input))  # 第二个全连接层
        input = self.drop_3(input)
        input = self.fc3(input)  # 输出层

        return input
