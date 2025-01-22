# End2EndAD

## Models

|     Branch      |         Model        |
|-----------------|----------------------|  
|    `resnet`     |use resnet train AS NN|
|      `cnn`      |   use cnn train AS NN|
|      `lstm`     |use cnn+lstm train AS NN|
| `to be added...`|   to be added...     |

## Datasets
All data is collected using the "CitySample" simulator.

You can download the dataset at:  
[Saved_ordered.zip](https://syncandshare.lrz.de/getlink/fiP41JGQFZC5zofBEQwJ5a/Saved_ordered.zip)

### Contents
The dataset contains three scenarios: **go_straight**, **turning_right**, **turning_left**.  
Each scenario includes the following folders:

| Folder Name | Description                                                                                     | Example Data |
|-------------|-------------------------------------------------------------------------------------------------|--------------|
| `drivings`  | Contains car status information (Throttle, Brake, Steering, Wheel Angle, Heading, Position, Speed, Acceleration, Direction). Each timestamp is stored in a CSV file named `yyyymmdd_hhmmss_mmm`. | **Filename:** `20241103_170827_000.csv`<br>**Content:**<br>Throttle: 0<br>Brake: 0<br>Steering: 1<br>Wheel Angle: 6.058597<br>Heading: -171.15765<br>Position: X=-49698.427 Y=13564.725 Z=3.951<br>Speed: 18.020468<br>Acc: -0.562561<br>Direction: Forward |
| `images`    | Contains left, mid, and right camera images. Each timestamp is stored as a `.png` image named `yyyymmdd_hhmmss_mmm_(L/M/R)` (L/M/R are camera tags). | **Filename:** `20241103_170801_000_M.png` |
| `labels`    | Contains relevant labels for objects in the images. Each timestamp is stored in a CSV file named `yyyymmdd_hhmmss_mmm_(L/M/R)` (L/M/R are camera tags). | **Filename:** `20241103_172137_000_L.csv`<br>**Content:**<br>car: 177, 175, 189, 184<br>car: 128, 169, 161, 194<br>Human: 0, 171, 8, 186 |

### Data Naming Convention
All data is named as `yyyymmdd_hhmmss_mmm` or `yyyymmdd_hhmmss_mmm_(L/M/R)` where:
- `yyyymmdd`: Date in the format Year-Month-Day.
- `hhmmss`: Time in the format Hour-Minute-Second.
- `mmm`: Milliseconds.
- `(L/M/R)`: Camera tags representing Left, Mid, and Right cameras.
