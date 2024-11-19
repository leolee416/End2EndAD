import os # Path operation
import glob # To find the path of specific files(eg. *.png)
import pandas as pd # .csv operation
from PIL import Image # Pillow(PIL) load image
from torch.utils.data.dataset import Dataset # basic data class
import torchvision.transforms as transform # size adjustance and normalization
import torch # introduce Tensor, store datas in tensor
import logging
from tqdm import tqdm 


class SimulatorDataset(Dataset):
    def __init__(self, image_dir, label_dir, transform = None, output_fields = None):
        """
        :param image_dir: path to folder that storing images 
        :param label_dir: path to folder that sotring labels
        :param transform : flag of image preprocessing and augumentation
        :param output_fields: outputs of Network(eg. ["steering"] or ["Steering", "Throttle", "Brake"])
        """

        self.image_dir = image_dir
        self.label_dir = label_dir
        self.transform = transform
        self.output_fields = output_fields if output_fields else ["Steering"] # default output

        # get the path of every .PNG under "image_dir"
        self.image_files = glob.glob(os.path.join(image_dir, "*.png")) 
        self.image_files.sort() # sort the files by time stamp

        # Call the private METHOD data_map to map the image with corresponding labels
        self.data_map = self._create_data_map()  
    
    # def _create_data_map(self):
    #     """
    #     Analyze the time stamps, create the map between images and labels.
    #     Logs progress for tracking.
    #     """
    #     # old version
    #     data_map = []
    #     for img_path in self.image_files:
    #         filename = os.path.basename(img_path)
    #         parts = filename.split('_')
    #         timestamp = "_".join(parts[:3])
    #         camera = parts[-1].split('.')[0]
        
    #         # Find corresponding label
    #         label_path = os.path.join(self.label_dir, f"{timestamp}.csv")
    #         if os.path.exists(label_path):
    #             data_map.append({"image": img_path, "label": label_path, "camera": camera})
    #     return data_map

    #     # try optimize : preload the files name of label_path, accelerate the reading speed
    def _create_data_map(self):
        """
        Analyze the time stamps, create the map between images and labels.
        Displays progress bar using tqdm and logs the number of successful mappings.
        """
        # Create a dictionary of label files indexed by timestamp
        label_files = {os.path.basename(f).split('.')[0]: f for f in glob.glob(os.path.join(self.label_dir, "*.csv"))}
        data_map = []

        # Progress bar for mapping process
        for img_path in tqdm(self.image_files, desc="Mapping Images to Labels", unit="file"):
            filename = os.path.basename(img_path)
            parts = filename.split('_')
            timestamp = "_".join(parts[:3])  # Extract timestamp
            camera = parts[-1].split('.')[0]  # Extract camera tag
            
            # Find corresponding label
            if timestamp in label_files:
                label_path = label_files[timestamp]
                data_map.append({"image": img_path, "label": label_path, "camera": camera})

        # Log the count of successfully matched image-label pairs
        matched_count = len(data_map)
        print(f"Successfully matched {matched_count} image-label pairs.")
        logging.info(f"Successfully matched {matched_count} image-label pairs.")

        return data_map

    def _parse_label(self, label_path) :
        """
        Analyse the .csv label file, assign Groundtruth to label_data
        """
        df = pd.read_csv(label_path,  header = None, index_col= 0)
        label_data = {}
        for key, value in df[1].items():
            key = key.strip()
            if key == "Position":
                value = value.replace('X=','').replace('Y=','').replace('Z=','')
                label_data[key] = [float(v) for v in value]
            elif key in ["Throttle","Brake","Steering","Wheel Angle","Heading","Speed","Acc"]:
                label_data[key] = float(value)
            elif key == "Direction":
                label_data[key] = 1 if value == "forward" else 0 # convert forward & backward to 1 & 0
            else:
                label_data[key] = value

        return label_data
    
    
    def __len__(self):
        return len(self.data_map)

    def __getitem__(self, index) :
        item_data = self.data_map[index]
        img_path = item_data["image"]
        label_path = item_data["label"]
        cam_tag = item_data["camera"]

        image = Image.open(img_path)
        if self.transform:
            image = self.transform(image)

        label_data = self._parse_label(label_path)
        
        # manually add offset to steering angles 
        if cam_tag == "L":
            label_data["Steering"] += 0.2
        elif cam_tag == "R":
            label_data["Steering"] -= 0.2

        # the steering angle must within [-1, 1]
        label_data["Steering"] = max(-1, min(1, label_data["Steering"]))
        
        output_labels = torch.tensor([label_data[field] for field in self.output_fields], dtype= torch.float32)

        return image, output_labels
    
    
    