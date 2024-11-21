# End2EndAD


## Intro
The recommanded workspace structure is(the files name just for demonstration):  
workspace/  
│  
├──Dataset/               # 数据存放目录  
│   ├── drivings/          # 驾驶行为标签文件  
│   │   ├── driving_label_001.csv  
│   │   ├── driving_label_002.csv  
│   │   └── ...  
│   └── images/            # 图像数据文件  
│       ├── image_001.jpg  
│       ├── image_002.jpg  
│       └── ...  
│  
├── data_loader.py         # 数据加载模块  
├── network_model.py       # 神经网络模型定义  
├── End2End_ResNet.ipynb   # 用于端到端训练和实验的 Jupyter Notebook  
├── drive.py               # 驾驶测试代码，用于运行预测模型  
├── network_model_for_run.py # 精简版模型定义，供推理时使用  
└── NNServer.py            # 神经网络服务器脚本  





### 说明

#### Dataset/
- **drivings/**:
  - 存储驾驶数据标签，文件以 `.csv` 格式组织。
- **images/**:
  - 存储与驾驶数据标签对应的图像文件。

#### 代码文件
- **data_loader.py**:
  - 负责加载并预处理图像及标签数据。
- **network_model.py**:
  - 定义深度学习模型结构，用于训练和测试。
- **End2End_ResNet.ipynb**:
  - 提供端到端模型训练与实验的详细步骤。
- **drive.py**:
  - 使用训练好的模型进行推理，生成驾驶预测结果。
- **network_model_for_run.py**:
  - 为推理优化的模型版本，减少不必要的依赖。
- **NNServer.py**:
  - 基于模型的服务器端实现，可通过 API 提供推理服务。

## Pipelines

### Train Pipeline
- **End2End_ResNet.ipynb**:
  - 用于端到端训练模型，包含数据加载、网络初始化和训练流程。
- **network_model.py**:
  - 可通过修改该文件调整神经网络结构，例如改变层数或激活函数。
- **data_loader.py**:
  - 负责加载数据集（包括图像和标签），并将其封装为训练所需的对象，支持批量加载和预处理操作。

### Evaluation Pipeline
- **NNServer.py**:
  - 实现一个服务器端，用于接收来自模拟器的实时图像，处理后返回控制信号。
- **drive.py**:
  - 包含一个预测类，能够接收输入图像并生成控制信号，例如转向、加速或减速。
- **network_model_for_run.py**
  - 包含神经网络结构

### Training

## Datasets
All data is collected using the "CitySample" simulator.

You can download the dataset at:  
[Saved_ordered.zip](https://syncandshare.lrz.de/getlink/fiP41JGQFZC5zofBEQwJ5a/Saved_ordered.zip)

### Contents
The dataset contains three scenarios: **go_straight**, **turning_right**, **turning_left**.  
Each scenario includes the following folders:

| Folder Name | Description                                                                                     | Example Data |
|-------------|-------------------------------------------------------------------------------------------------|--------------|
| `drivings`  | Contains car status information (Throttle, Brake, Steering, Wheel Angle, Heading, Position, Speed, Acceleration, Direction). Each timestamp is stored in a CSV file named `yyyymmdd_hhmmss_mmm`. | **Filename:** `20241103_170827_000.csv`<br>**Content:**<br>Throttle: 0<br>Brake: 0<br>Steering: 1(clock-wise)<br>Wheel Angle: 6.058597(clock-wise)<br>Heading: -171.15765<br>Position: X=-49698.427 Y=13564.725 Z=3.951<br>Speed: 18.020468<br>Acc: -0.562561<br>Direction: Forward |
| `images`    | Contains left, mid, and right camera images. Each timestamp is stored as a `.png` image named `yyyymmdd_hhmmss_mmm_(L/M/R)` (L/M/R are camera tags). | **Filename:** `20241103_170801_000_M.png` |
| `labels`    | Contains relevant labels for objects in the images. Each timestamp is stored in a CSV file named `yyyymmdd_hhmmss_mmm_(L/M/R)` (L/M/R are camera tags). | **Filename:** `20241103_172137_000_L.csv`<br>**Content:**<br>car: 177, 175, 189, 184<br>car: 128, 169, 161, 194<br>Human: 0, 171, 8, 186 |

### Data Naming Convention
All data is named as `yyyymmdd_hhmmss_mmm` or `yyyymmdd_hhmmss_mmm_(L/M/R)` where:
- `yyyymmdd`: Date in the format Year-Month-Day.
- `hhmmss`: Time in the format Hour-Minute-Second.
- `mmm`: Milliseconds.
- `(L/M/R)`: Camera tags representing Left, Mid, and Right cameras.
