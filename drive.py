import NNServer
import numpy as np
import struct
from torchvision import transforms
from network_model import ModelCNN
import torch
from PIL import Image

class AutoDrive:
    def __init__(self, model_path):
        self.model = self.load_model(model_path)
        self.transforms = transforms.Compose([
            transforms.Resize((224, 224)),  # 模型输入大小
            transforms.ToTensor(),         # 转为 PyTorch 张量
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # 归一化
        ])

    def load_model(self, model_path):
        """
        加载模型参数，支持两种情况：
        1. 提供仅包含模型参数的 state_dict 文件。
        2. 提供完整的训练检查点文件。
        """
        model = ModelCNN()  # 初始化你的模型
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # 加载文件
        checkpoint = torch.load(model_path, map_location=torch.device, weights_only= True)  

        # 检查文件内容，判断是 state_dict 还是完整检查点
        if 'model_state_dict' in checkpoint:  # 检查点文件
            print("[INFO] Loading from checkpoint...")
            model.load_state_dict(checkpoint['model_state_dict'])
        elif isinstance(checkpoint, dict):  # 仅包含 state_dict 的权重文件
            print("[INFO] Loading from state_dict...")
            model.load_state_dict(checkpoint)
        else:
            raise ValueError("[ERROR] Unsupported file format. Ensure the file is either a state_dict or a checkpoint.")

        model.eval()  # 设置为评估模式
        print("[INFO] Model loaded successfully!")
        return model
    
    def preprocess_image(self, image_data):
        """
        preprocesse the image stream:
        Normalization, transpose
        """
        image = Image.fromarray(image_data)

        image_tensor = self.transforms(image)
        return image_tensor.unsqueeze(0)
    
    def predict_control(self, image):
        with torch.no_grad():
            image_tensor = self.preprocess_image(image)
            output = self.model(image_tensor)
            Steering, Throttle, Brake = output[0].numpy()
        print("[INFO] Prediction: Steering = {Steering}, Throttle = {Throttle}, Brake = {Brake}")
        return Steering, Throttle, Brake
    
def process_client_data(data_package, auto_dirve):
    """
    accept stream data and autodrive object.
    Send control data to cclient
    """
    IMAGE_W = NNServer.IMAGE_WIDTH
    IMAGE_H = NNServer.IMAGE_HEIGHT
    IMAGE_C = NNServer.IMAGE_CHANNELS
            
    # Get the second image(Mid camera image)
    second_image_start = IMAGE_W * IMAGE_H * IMAGE_C # all images are combined into 3*H x 3*W x 3 "big picture"
    second_image_end = 2 * second_image_start
    second_image_data = data_package.ImageData[second_image_start:second_image_end]

    # uncoded in Numpy array
    second_image = np.frombuffer(second_image_data, dtype=np.uint8).reshape(IMAGE_H, IMAGE_W, IMAGE_C)

    steering, throttle, brake = auto_drive.predict_control(second_image)
    return steering, throttle, brake
