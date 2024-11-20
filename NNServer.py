import socket
import struct
import threading
import numpy as np
import cv2

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

# 显示图片
def show_image(data):
    combined_height = IMAGE_HEIGHT * 3
    combined_width = IMAGE_WIDTH * 3
    combined_image = np.zeros((combined_height, combined_width, IMAGE_CHANNELS), dtype=np.uint8)

    positions = [
        (0, 0), (IMAGE_WIDTH, 0), (2 * IMAGE_WIDTH, 0),  # 第一行
        (IMAGE_WIDTH // 2, IMAGE_HEIGHT), (IMAGE_WIDTH + IMAGE_WIDTH // 2, IMAGE_HEIGHT)  # 第二行
    ]

    for i in range(NUM_IMAGES):
        image_data = np.frombuffer(data[i * IMAGE_WIDTH * IMAGE_HEIGHT * IMAGE_CHANNELS:(i + 1) * IMAGE_WIDTH * IMAGE_HEIGHT * IMAGE_CHANNELS], dtype=np.uint8)
        image = image_data.reshape((IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS))
        x, y = positions[i]
        combined_image[y:y + IMAGE_HEIGHT, x:x + IMAGE_WIDTH] = image

    cv2.imshow("Combined Image", combined_image)
    cv2.waitKey(1)

# 处理数据
def process_data(data_package):
    show_image(data_package.ImageData)
    print(data_package.Direction, data_package.Position.X, data_package.Position.Y, data_package.Position.Z)

# 反序列化数据包
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

# 处理客户端连接
def handle_client(client_socket):
    recvbuf = bytearray(DEFAULT_BUFLEN)

    while True:
        try:
            bytes_received = client_socket.recv_into(recvbuf)
            if bytes_received > 0:
                data_package = deserialize_vehicle_data(recvbuf)
                total_size = IMAGE_WIDTH * IMAGE_HEIGHT * IMAGE_CHANNELS * NUM_IMAGES
                if len(data_package.ImageData) != total_size:
                    print(f"Data size mismatch: expected {total_size} bytes, received {len(recvbuf)} bytes")
                    response = f"Data size mismatch: expected {total_size} bytes, received {len(recvbuf)} bytes"
                    client_socket.sendall(response.encode('utf-8'))
                else:
                    process_data(data_package)

                # 发送响应 (throttle;brake;steering)
                response = "1;0;0"
                client_socket.sendall(response.encode('utf-8'))
            else:
                print("Connection closing...")
                break
        except Exception as e:
            print(f"Exception: {e}")
            break

    client_socket.close()

# 主函数
def main():
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind(('0.0.0.0', DEFAULT_PORT))
    server_socket.listen()

    print(f"Waiting for client connection on port {DEFAULT_PORT}...")

    while True:
        client_socket, addr = server_socket.accept()
        print(f"Client connected from {addr}")

        client_handler = threading.Thread(target=handle_client, args=(client_socket,))
        client_handler.start()

if __name__ == "__main__":
    main()
