import NNServer_2
import numpy as np
import struct
from torchvision import transforms
from network_model_for_run import ModelCNN
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
        Load the model to the specified device (GPU or CPU)
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device

        model = ModelCNN()
        try:
            checkpoint = torch.load(model_path, map_location=device, weights_only= True)
            if 'model_state_dict' in checkpoint:
                print("[INFO] Loading from checkpoint...")
                model.load_state_dict(checkpoint['model_state_dict'])
            elif isinstance(checkpoint, dict):
                print("[INFO] Loading from state_dict...")
                model.load_state_dict(checkpoint)
            else:
                raise ValueError("[ERROR] Unsupported file format.")
            model.to(device)  # 模型放置到设备
            model.eval()
            print(f"[INFO] Model loaded successfully on {device}!")
        except Exception as e:
            raise RuntimeError(f"[ERROR] Failed to load model: {e}")
        return model
    
    def preprocess_image(self, image_data):
        """
        Preprocess the image stream:
        - Resize
        - Normalize
        - Convert to the same device as the model
        """
        try:
            image = Image.fromarray(image_data)
            image_tensor = self.transforms(image).to(self.device)  # 将数据移动到模型所在设备
            return image_tensor.unsqueeze(0)  # 增加 batch 维度
        except Exception as e:
            print(f"[ERROR] Preprocessing failed: {e}")
            return None
    
    def predict_control(self, image):
        try:
            image_tensor = self.preprocess_image(image)
            if image_tensor is None:
                return 0.0, 0.0, 1.0  # 默认值

            with torch.no_grad():
                output = self.model(image_tensor)  # 获取模型输出
                
                # 检查输出是否包含三个值
                if output.shape[-1] != 3:
                    raise ValueError(f"Unexpected output shape: {output.shape}. Expected shape with 3 values.")

                steering, throttle, brake = output[0].cpu().numpy()  # 输出转移到 CPU
            return steering, throttle, brake
        except Exception as e:
            print(f"[ERROR] Prediction failed: {e}")
            return 0.0, 0.0, 1.0  # 默认值

    
def process_client_data(data_package, auto_dirve):
    """
    accept stream data and autodrive object.
    Send control data to cclient
    """
    IMAGE_W = NNServer_2.IMAGE_WIDTH
    IMAGE_H = NNServer_2.IMAGE_HEIGHT
    IMAGE_C = NNServer_2.IMAGE_CHANNELS
            
    # Get the second image(Mid camera image)
    second_image_start = IMAGE_W * IMAGE_H * IMAGE_C # all images are combined into 3*H x 3*W x 3 "big picture"
    second_image_end = 2 * second_image_start
    second_image_data = data_package.ImageData[second_image_start:second_image_end]

    # uncoded in Numpy array
    second_image = np.frombuffer(second_image_data, dtype=np.uint8).reshape(IMAGE_H, IMAGE_W, IMAGE_C)

    steering, throttle, brake = auto_drive.predict_control(second_image)
    return steering, throttle, brake
