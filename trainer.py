import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import matplotlib.pyplot as plt

class Trainer:
    def __init__(self, model, train_dataset, val_dataset, config):
        """
        初始化 Trainer 类。
        :param model: PyTorch 模型实例
        :param train_dataset: 训练集
        :param val_dataset: 验证集
        :param config: 配置字典，包含训练参数
        """
        self.model = model.cuda() if torch.cuda.is_available() else model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset

        # 配置训练参数
        self.batch_size = config.get("batch_size", 32)
        self.num_epochs = config.get("num_epochs", 10)
        self.learning_rate = config.get("learning_rate", 1e-4)
        self.samples_per_epoch = config.get("samples_per_epoch", len(train_dataset))
        self.model_dir = config.get("model_dir", "models")
        self.log_dir = config.get("log_dir", "logs")

        # 初始化损失函数和优化器
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

        # TensorBoard writer
        self.writer = SummaryWriter(log_dir=self.log_dir)

        # DataLoader
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
        )
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
        )

        # 用于存储训练和验证损失
        self.train_losses = []
        self.val_losses = []
        self.train_steps = []
        self.val_steps = []

    def train_one_epoch(self, epoch):
        """训练一个 epoch"""
        self.model.train()
        total_loss = 0
        for step, (images, labels) in enumerate(tqdm(self.train_loader, desc=f"Epoch {epoch}", leave=False)):
            images, labels = images.cuda(), labels.cuda()

            # 前向传播
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)

            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

        # 计算当前 epoch 的平均损失
        avg_loss = total_loss / len(self.train_loader)

        # 记录每个 epoch 的训练损失到 TensorBoard
        self.writer.add_scalar('Train/Loss', avg_loss, epoch)

        return avg_loss

    def validate(self):
        """验证模型"""
        self.model.eval()
        total_loss = 0
        with torch.no_grad():
            for images, labels in tqdm(self.val_loader, desc="Validating"):
                images, labels = images.cuda(), labels.cuda()  # 数据迁移到 GPU

                # 前向传播
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                total_loss += loss.item()

        return total_loss / len(self.val_loader)

    def train(self):
        """主训练逻辑，记录每个 epoch 的训练和验证损失"""
        for epoch in range(self.num_epochs):
            # 训练一个 epoch
            train_loss = self.train_one_epoch(epoch)
            
            # 验证模型
            val_loss = self.validate()

            # 记录验证损失到 TensorBoard
            self.writer.add_scalar('Validation/Loss', val_loss, epoch)

            # 打印训练和验证损失
            print(f"Epoch {epoch + 1}/{self.num_epochs} | Train Loss: {train_loss:.8f} | Val Loss: {val_loss:.8f}")

            # 保存模型
            os.makedirs(self.model_dir, exist_ok=True)
            model_path = os.path.join(self.model_dir, f"model_epoch_{epoch + 1}.pth")
            torch.save(self.model.state_dict(), model_path)

            # 每个 epoch 绘制损失曲线
            self.plot_losses()

    def plot_losses(self):
        """绘制训练和验证损失曲线"""
        plt.figure(figsize=(10, 5))
        plt.plot(self.train_steps, self.train_losses, label="Train Loss")
        plt.plot(self.val_steps, self.val_losses, label="Validation Loss")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.title("Training and Validation Loss")
        plt.grid()
        plt.savefig("loss_curve.png")
        plt.show()
