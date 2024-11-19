import os  # Path operation
import glob  # To find the path of specific files (e.g., *.png)
import pandas as pd  # .csv operation
from PIL import Image  # Pillow (PIL) load image
from torch.utils.data.dataset import Dataset  # Basic data class
import torchvision.transforms as transform  # Size adjustment and normalization
import torch  # Introduce Tensor, store data in tensor
from tqdm import tqdm

class SimulatorDataset(Dataset):
    def __init__(self, image_dir, label_dir, transform=None, output_fields=None):
        """
        :param image_dir: Path to folder storing images 
        :param label_dir: Path to folder storing labels
        :param transform: Image preprocessing and augmentation
        :param output_fields: Outputs of the network (e.g., ["Steering"] or ["Steering", "Throttle", "Brake"])
        """

        self.image_dir = image_dir
        self.label_dir = label_dir
        self.transform = transform
        self.output_fields = output_fields if output_fields else ["Steering"]  # Default output

        # Get the path of every .PNG under "image_dir"
        self.image_files = glob.glob(os.path.join(image_dir, "*.PNG"))
        self.image_files.sort()  # Sort the files by timestamp

        # Print number of image files found
        print(f"Found {len(self.image_files)} image files in {self.image_dir}.")

        # Call the private method to map images with corresponding labels
        self.data_map = self._create_data_map()
        
        # Print number of successful mappings
        print(f"Created data_map with {len(self.data_map)} entries.")

    def _create_data_map(self):
        """
        Analyze the time stamps, create the map between images and labels.
        Displays progress bar using tqdm and logs debug information.
        """
        label_files = {os.path.basename(f).split('.')[0]: f for f in glob.glob(os.path.join(self.label_dir, "*.csv"))}
        data_map = []

        # Print number of label files found
        print(f"Found {len(label_files)} label files in {self.label_dir}.")

        for img_path in tqdm(self.image_files, desc="Mapping Images to Labels", unit="file"):
            filename = os.path.basename(img_path)
            parts = filename.split('_')
            timestamp = "_".join(parts[:3])  # Extract timestamp
            camera = parts[-1].split('.')[0]  # Extract camera tag

            # Find corresponding label
            if timestamp in label_files:
                label_path = label_files[timestamp]
                data_map.append({"image": img_path, "label": label_path, "camera": camera})
            else:
                # Print unmatched image files
                print(f"No label found for image {filename}.")

        # Print the number of successful mappings
        matched_count = len(data_map)
        print(f"Successfully matched {matched_count} image-label pairs.")

        return data_map

    def _parse_label(self, label_path):
        """
        Analyze the .csv label file, assign ground truth to label_data.
        """
        def _clean_float_value(value, key, label_path):
            """
            Helper function to clean and convert a value to float.
            Handles special cases like Unicode negative signs or invalid formats.
            """
            try:
                # 将值转换为字符串，替换特殊负号，并尝试转换为浮点数
                value = str(value).replace("−", "-").strip()  # 替换 Unicode 负号并去掉两边空格
                return float(value)
            except ValueError:
                print(f"Warning: Invalid float value for key '{key}' in file '{label_path}': {value}. Setting to 0.0.")
                return 0.0

        df = pd.read_csv(label_path, header=None, index_col=0)
        label_data = {}
        for key, value in df[1].items():
            key = key.strip()
            if key == "Position":
                try:
                    # 清理 Position 数据，确保可以分解成 float
                    value = value.replace('X=', '').replace('Y=', '').replace('Z=', '').strip()
                    label_data[key] = [float(v.strip()) for v in value.split()]  # 清理并分割 Position 数据
                except ValueError as e:
                    print(f"Error parsing Position in file '{label_path}': {value}. Error: {e}")
                    label_data[key] = [0.0, 0.0, 0.0]  # 设置默认值
            elif key in ["Throttle", "Brake", "Steering", "Wheel Angle", "Heading", "Speed", "Acc"]:
                label_data[key] = _clean_float_value(value, key, label_path)
            elif key == "Direction":
                label_data[key] = 1 if value.strip().lower() == "forward" else 0  # Convert forward & backward to 1 & 0
            else:
                label_data[key] = value.strip()  # 对其他字段进行简单清理

        # Print parsed label data for debugging
        print(f"Parsed labels from {label_path}: {label_data}")

        return label_data


    def __len__(self):
        # Print dataset length
        dataset_length = len(self.data_map)
        print(f"Dataset length: {dataset_length}")
        return dataset_length

    def __getitem__(self, index):
        # 获取当前索引的数据
        item_data = self.data_map[index]
        img_path = item_data["image"]
        label_path = item_data["label"]
        cam_tag = item_data["camera"]

        # 检查图像路径是否存在
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Image file not found at: {img_path}")

        # 加载图像并确保是 RGB 格式
        image = Image.open(img_path).convert("RGB")
        print(f"Loaded image at index {index}: {img_path} (Size: {image.size})")

        # 应用预处理（如调整大小、转换为张量、归一化）
        if self.transform:
            image = self.transform(image)
        print(f"successfully transform the image")

        # 检查是否为单通道，若是则转换为 3 通道
        if isinstance(image, torch.Tensor) and image.size(0) == 1:
            image = image.repeat(3, 1, 1)
        print(f"check the channels")

        # 检查处理后的图像类型和形状
        if not isinstance(image, torch.Tensor):
            raise TypeError(f"Transformed image is not a Tensor. Got: {type(image)}")
        print(f"Transformed image shape: {image.shape}")

        # 解析标签
        label_data = self._parse_label(label_path)

        # 根据摄像头角度调整转向角
        if cam_tag == "L":
            label_data["Steering"] += 0.2
        elif cam_tag == "R":
            label_data["Steering"] -= 0.2

        # 限制转向角在 [-1, 1]
        label_data["Steering"] = max(-1, min(1, label_data["Steering"]))

        # 提取指定的输出字段并转换为 Tensor
        try:
            output_labels = torch.tensor([label_data[field] for field in self.output_fields], dtype=torch.float32)
        except KeyError as e:
            raise KeyError(f"Field {e} is missing in label data: {label_data}")

        # 输出调试信息
        print(f"Output labels for index {index}: {output_labels.tolist()}")

        return image, output_labels

