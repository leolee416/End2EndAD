import torch
import torchvision.models as models
import torch.nn as nn

class ModelCNN(nn.Module):
    def __init__(self, output_fields=None, use_pretrained=True):
        """
        初始化模型
        :param output_fields: 要预测的输出字段列表
        :param use_pretrained: 是否加载预训练权重
        """
        super().__init__()

        # 设备配置
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 激活函数
        self.elu = nn.ELU()

        # 加载 ResNet50
        Resnet50 = models.resnet50(weights=models.ResNet50_Weights.DEFAULT if use_pretrained else None)
        self.model = torch.nn.Sequential(*list(Resnet50.children())[:-3])  # 截断 ResNet50

        # Dropout 层
        self.drop_1 = nn.Dropout(0.5)
        self.Flatten = nn.Flatten()

        # 输出字段
        self.output_fields = output_fields if output_fields else ["Steering"]
        self.num_outputs = len(self.output_fields)

        # 全连接层在 forward 中动态初始化
        self.fc1 = None  
        self.drop_2 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(100, 50)  # 第二个全连接层
        self.drop_3 = nn.Dropout(0.5)
        self.fc3 = None  # 动态初始化输出层

    def initialize_fc_layer(self, layer_name, in_features, out_features):
        """
        动态初始化全连接层
        """
        layer = getattr(self, layer_name)
        if layer is None:
            layer = nn.Linear(in_features, out_features).to(self.device)
            setattr(self, layer_name, layer)
        return layer

    def forward(self, input):
        # 通过截断的 ResNet50
        input = self.model(input)
        input = self.drop_1(input)
        input = self.Flatten(input)

        # 动态初始化 fc1 和 fc3
        self.fc1 = self.initialize_fc_layer("fc1", input.size(1), 100)
        self.fc3 = self.initialize_fc_layer("fc3", 50, self.num_outputs)

        # 通过全连接层
        input = self.elu(self.fc1(input))  # 第一个全连接层
        input = self.drop_2(input)
        input = self.elu(self.fc2(input))  # 第二个全连接层
        input = self.drop_3(input)
        input = self.fc3(input)  # 输出层

        return input

    def load_state_dict(self, state_dict, strict=False):
        """
        自定义加载 state_dict，支持动态初始化的层
        """
        missing_keys, unexpected_keys = super().load_state_dict(state_dict, strict=strict)
        if missing_keys:
            print(f"[WARNING] Missing keys during state_dict loading: {missing_keys}")
        if unexpected_keys:
            print(f"[WARNING] Unexpected keys during state_dict loading: {unexpected_keys}")
