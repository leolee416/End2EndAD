import torch
import torchvision.models as models
import torch.nn as nn

class model_cnn(nn.Module):
    def __init__(self):
        super().__init__()

        self.elu = nn.ELU()  # 激活函数 ELU
        Resnet50 = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)  # 加载预训练 ResNet50
        self.model = torch.nn.Sequential(*list(Resnet50.children())[:-3])  # 移除 layer4, avgpool 和 fc
        self.drop_1 = nn.Dropout(0.5)  # Dropout 层
        self.Flatten = nn.Flatten()  # 展平层

        # 全连接层在 forward 中动态初始化
        self.fc1 = None  
        self.drop_2 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(100, 50)  # 第二个全连接层
        self.drop_3 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(50, 1)  # 输出转向角

    def forward(self, input):
        input = input / 255.0  # 归一化处理
        input = self.model(input)  # 通过截断的 ResNet50
        input = self.drop_1(input)
        input = self.Flatten(input)  # 展平

        # 动态计算展平后的特征维度并初始化 fc1
        if self.fc1 is None:
            in_features = input.size(1)  # 展平后的特征数
            self.fc1 = nn.Linear(in_features, 100).to(input.device)

        input = self.elu(self.fc1(input))  # 第一个全连接层
        input = self.drop_2(input)
        input = self.elu(self.fc2(input))  # 第二个全连接层
        input = self.drop_3(input)
        input = self.fc3(input)  # 输出层
        return input

# 测试模型
if __name__ == "__main__":
    model = model_cnn()
    dummy_input = torch.randn(1, 3, 224, 224)  # 模拟一个输入张量
    output = model(dummy_input)  # 检查输出
    print(f"Output shape: {output.shape}")
