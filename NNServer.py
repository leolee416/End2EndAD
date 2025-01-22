import socket
import struct
import threading
import numpy as np
import cv2
import torch
from model_lstm import ComplexLSTM
from PIL import Image
import time

# 定义常量
DEFAULT_PORT = 12345
IMAGE_WIDTH = 640
IMAGE_HEIGHT = 360
IMAGE_CHANNELS = 3
NUM_IMAGES = 3
DEFAULT_BUFLEN = IMAGE_WIDTH * IMAGE_HEIGHT * IMAGE_CHANNELS * NUM_IMAGES + 1024

# FVector 结构
class FVector:
    def __init__(self, x=0.0, y=0.0, z=0.0):
        self.X = x
        self.Y = y
        self.Z = z

# DataPackage 结构
class DataPackage:
    def __init__(self):
        self.Steering = 0.0
        self.Throttle = 0.0
        self.Brake = 0.0
        self.WheelAngle = 0.0
        self.Timestamp = 0.0
        self.Acceleration = 0.0
        self.Direction = 0
        self.Position = FVector()
        self.Velocity = 0.0
        self.Heading = 0.0
        self.ImageData = bytearray()

def extract_second_camera_sequence(data_package_list):
    sequence = []
    for idx, data_package in enumerate(data_package_list):
        second_image_start = 1 * IMAGE_WIDTH * IMAGE_HEIGHT * IMAGE_CHANNELS
        second_image_end = second_image_start + IMAGE_WIDTH * IMAGE_HEIGHT * IMAGE_CHANNELS
        second_image_data = data_package.ImageData[second_image_start:second_image_end]

        # 检查数据长度是否正确
        assert len(second_image_data) == IMAGE_WIDTH * IMAGE_HEIGHT * IMAGE_CHANNELS, \
            f"Data length mismatch for package {idx}: {len(second_image_data)}"

        second_image = np.frombuffer(second_image_data, dtype=np.uint8).reshape(
            IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS)
        sequence.append(second_image)
    return sequence


def get_velocity_sequence(data_package_list):
    velocities = [data_package.Velocity for data_package in data_package_list]
    return torch.tensor(velocities, dtype=torch.float32).unsqueeze(0).unsqueeze(-1)

def preprocess_sequence(sequence, device):
    processed_sequence = []
    for image in sequence:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (200, 66))
        image = image / 255.0
        image_tensor = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1)
        processed_sequence.append(image_tensor)
    return torch.stack(processed_sequence).unsqueeze(0).to(device)

def deserialize_vehicle_data(data):
    package = DataPackage()
    offset = 0

    total_image_size = IMAGE_WIDTH * IMAGE_HEIGHT * IMAGE_CHANNELS * NUM_IMAGES
    package.ImageData = data[:total_image_size]
    offset += total_image_size

    package.Position = FVector(*struct.unpack('ddd', data[offset:offset + 24]))
    offset += 24

    package.Velocity, = struct.unpack('f', data[offset:offset + 4])
    offset += 4

    package.Heading, = struct.unpack('f', data[offset:offset + 4])
    offset += 4

    package.Direction, = struct.unpack('i', data[offset:offset + 4])
    offset += 4

    package.Acceleration, = struct.unpack('f', data[offset:offset + 4])
    offset += 4

    package.Timestamp, = struct.unpack('f', data[offset:offset + 4])
    offset += 4

    package.WheelAngle, = struct.unpack('f', data[offset:offset + 4])
    offset += 4

    package.Brake, = struct.unpack('f', data[offset:offset + 4])
    offset += 4

    package.Throttle, = struct.unpack('f', data[offset:offset + 4])
    offset += 4

    package.Steering, = struct.unpack('f', data[offset:offset + 4])

    return package


def handle_client(client_socket, complex_lstm, device, sequence_length=10):
    recvbuf = bytearray(DEFAULT_BUFLEN)
    data_package_list = []
    image_sequence = []

    while True:
        try:
            # 接收数据包
            bytes_received = client_socket.recv_into(recvbuf)
            if bytes_received > 0:
                data_package = deserialize_vehicle_data(recvbuf)
                data_package_list.append(data_package)

                # 提取第二张图片
                second_image = extract_second_camera_sequence([data_package])[0]
                image_sequence.append(second_image)
                # 检查序列长度
                if len(image_sequence) >= sequence_length:
                    processed_image_sequence = preprocess_sequence(image_sequence[-sequence_length:], device)
                    velocity_sequence = get_velocity_sequence(data_package_list[-sequence_length:]).to(device)
                    velocity_array = velocity_sequence.cpu().numpy().squeeze()
                    transposed_velocity = velocity_array.T
                    # print(f"[DEBUG] Transposed velocity sequence: {transposed_velocity}")
                    # 模型推理
                    with torch.no_grad():
                        output = complex_lstm(processed_image_sequence, velocity_sequence)

                    # 解析预测结果
                    throttle, brake, steering = output.squeeze(0).tolist()
                    # 打印原始预测结果
                    print(f"[DEBUG] Raw prediction results - Throttle: {throttle}, Brake: {brake}, Steering: {steering}")

                    # # 应用后处理逻辑
                    # if throttle - brake > 0:
                    #     throttle = throttle - brake
                    #     brake = 0
                    # else:
                    #     brake = abs(throttle - brake)
                    #     throttle = 0

                    # 打印后处理结果
                    print(f"[DEBUG] Post-processed results - Throttle: {throttle}, Brake: {brake}, Steering: {steering}")
                    print(f"throttle: {throttle}, brake: {brake}, steering: {steering}")

                    # 发送预测结果
                    response = f"{throttle};{brake};{steering}"
                    client_socket.sendall(response.encode('utf-8'))

                    # 清理历史数据
                    image_sequence = image_sequence[-sequence_length:]
                    data_package_list = data_package_list[-sequence_length:]
                else:
                    print(f"[INFO] Waiting for enough frames to fill the sequence... (Current: {len(image_sequence)}/{sequence_length})")
            else:
                print("Connection closing...")
                break
        except Exception as e:
            import traceback
            print(f"[ERROR] Exception occurred: {e}")
            traceback.print_exc()
            break

    client_socket.close()




def main():
    model_path = r'models\trained_lstm_model_left_3.pth'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    complex_lstm = ComplexLSTM().to(device)
    complex_lstm.load_state_dict(torch.load(model_path, map_location=device))
    complex_lstm.eval()

    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind(('0.0.0.0', DEFAULT_PORT))
    server_socket.listen()

    print(f"Waiting for client connection on port {DEFAULT_PORT}...")

    while True:
        client_socket, addr = server_socket.accept()
        print(f"Client connected from {addr}")

        client_handler = threading.Thread(target=handle_client, args=(client_socket, complex_lstm, device))
        client_handler.start()

if __name__ == "__main__":
    main()
