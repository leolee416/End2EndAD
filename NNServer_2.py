import socket
import struct
import threading
import numpy as np
import cv2
import signal
import sys

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
    print("[INFO] Processing data...")
    show_image(data_package.ImageData)
    print(f"[INFO] Vehicle state: Direction ={data_package.Direction}, Position={data_package.Position.X, data_package.Position.Y, data_package.Position.Z}")

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
    print("[INFO] Client handler started...")

    recvbuf = bytearray(DEFAULT_BUFLEN)

    while True:
        try:
            bytes_received = client_socket.recv_into(recvbuf)
            if bytes_received > 0:
                print(f"[INFO] Received {bytes_received} bytes from client.")
                data_package = deserialize_vehicle_data(recvbuf)
                total_size = IMAGE_WIDTH * IMAGE_HEIGHT * IMAGE_CHANNELS * NUM_IMAGES
                if len(data_package.ImageData) != total_size:
                    print(f"[ERROR] Data size mismatch: expected {total_size} bytes, received {len(recvbuf)} bytes")
                    response = f"Data size mismatch: expected {total_size} bytes, received {len(recvbuf)} bytes"
                    client_socket.sendall(response.encode('utf-8'))
                else:
                    print("[INFO] Successfully received image data.")
                    process_data(data_package)

                # 发送响应 (throttle;brake;steering)
                response = "1;0;0"
                client_socket.sendall(response.encode('utf-8'))
            else:
                print("[WARNING] Client disconnected!")
                break
        except Exception as e:
            print(f"[ERROR] Exception in client handler: {e}")
            break
        except KeyboardInterrupt:
            print("[INFO] Server shutting down ...")
            break

    client_socket.close()


class Server:
    def __init__(self, host='0.0.0.0', port=12345, buffer_size=1024):
        """
        初始化服务器。
        :param host: 服务器主机地址（默认是 0.0.0.0，监听所有接口）
        :param port: 服务器监听端口（默认是 12345）
        :param buffer_size: 数据缓冲区大小
        """
        self.host = host
        self.port = port
        self.buffer_size = buffer_size
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.bind((self.host, self.port))
        self.server_socket.listen()
        print(f"[INFO] Server started on {self.host}:{self.port}")

    def accept_client(self):
        """
        等待客户端连接。
        :return: 返回客户端 socket 和地址。
        """
        client_socket, addr = self.server_socket.accept()
        print(f"[INFO] Client connected from {addr}")
        return client_socket, addr

    def handle_client(self, client_socket, handler_function):
        """
        处理单个客户端连接。
        :param client_socket: 客户端 socket
        :param handler_function: 用于处理接收到的数据包的函数
        """
        with client_socket:
            while True:
                try:
                    # 接收数据
                    data = client_socket.recv(self.buffer_size)
                    if not data:
                        print("[WARNING] Client disconnected.")
                        break

                    # 调用处理函数
                    response = handler_function(data)
                    if response:
                        client_socket.sendall(response.encode('utf-8'))
                except Exception as e:
                    print(f"[ERROR] Exception: {e}")
                    break

    def run(self, handler_function):
        """
        启动服务器并处理连接。
        :param handler_function: 用于处理接收到的数据包的函数
        """
        print("[INFO] Server is running...")
        try:
            while True:
                client_socket, addr = self.accept_client()
                client_thread = threading.Thread(target=self.handle_client, args=(client_socket, handler_function))
                client_thread.start()
        except KeyboardInterrupt:
            print("[INFO] Server shutting down...")
        finally:
            self.server_socket.close()
# # 主函数
# def main():
#     try:
#         server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
#         server_socket.bind(('0.0.0.0', DEFAULT_PORT))
#         server_socket.listen()

#         print(f"[INFO] Waiting for client connection on port {DEFAULT_PORT}...")

#         while True:
#             try:
#                 client_socket, addr = server_socket.accept()
#                 print(f"Client connected from {addr}")

#                 client_handler = threading.Thread(target=handle_client, args=(client_socket,))
#                 client_handler.start()
#             except KeyboardInterrupt:
#                 print("[INFO] Server shutting down ...")
#                 break
#     except Exception as e:
#         print(f"[ERROR] Server error: {e}")
#     finally:
#         server_socket.close()

    

# if __name__ == "__main__":
#     main()
