 
import torch
import torch.nn as nn
import torch.nn.functional as F

class ComplexLSTM(nn.Module):
    def __init__(self):
        super(ComplexLSTM, self).__init__()

        # Define the convolutional layers for the image input
        self.conv1 = nn.Conv2d(3, 64, kernel_size=5, stride=2, padding=2)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=2)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=5, stride=2, padding=2)
        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1)
        self.conv5 = nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1)
        self.conv6 = nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1)

        # Batch Normalization after convolutional layers
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm2d(256)
        self.bn4 = nn.BatchNorm2d(512)
        self.bn5 = nn.BatchNorm2d(512)
        self.bn6 = nn.BatchNorm2d(512)

        # Fully connected layer for feature reduction
        self.fc1 = nn.Linear(512 * 2 * 4, 512)

        # Define LSTM for temporal processing
        self.lstm = nn.LSTM(input_size=512, hidden_size=256, num_layers=2, batch_first=True)

        # Fully connected layers for velocity input
        self.fc_speed = nn.Linear(1, 20)

        # Combine LSTM features and velocity features
        self.fc_combined = nn.Linear(256 + 20, 512)

        self.dropout = nn.Dropout(0.5)

        # Outputs
        self.fc_end = nn.Linear(512, 3)

    def forward(self, image_sequence, speed_sequence):
        batch_size, seq_length, channels, height, width = image_sequence.size()

        # Flatten sequence for CNN processing
        image_sequence = image_sequence.view(batch_size * seq_length, channels, height, width)

        # Process each image through CNN
        x = F.relu(self.bn1(self.conv1(image_sequence)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))
        x = F.relu(self.bn6(self.conv6(x)))

        x = x.reshape(x.size(0), -1)  # Flatten
        x = F.relu(self.fc1(x))  # Reduce dimensions

        # Reshape back to sequence format for LSTM
        x = x.view(batch_size, seq_length, -1)

        # Pass through LSTM
        x, _ = self.lstm(x)

        # Use the last output of the LSTM for classification
        x = x[:, -1, :]  # Shape: (batch_size, hidden_size)

        # Process velocity sequence
        speed_sequence = speed_sequence.view(batch_size, seq_length, -1)
        v = F.relu(self.fc_speed(speed_sequence))  # Apply fully connected layer on speed
        v = v.mean(dim=1)  # Aggregate velocity features over time

        # Combine LSTM and velocity features
        combined = torch.cat((x, v), dim=1)
        combined = F.relu(self.fc_combined(combined))

        # Compute the outputs (e.g., throttle, brake, steering)
        result = self.fc_end(combined)

        return result
'''
class ComplexCNN(nn.Module):
    def __init__(self):
        super(ComplexCNN, self).__init__()
        
        # Convolutional Layers with increasing depth
        self.conv1 = nn.Conv2d(3, 24, kernel_size=5, stride=2, padding=2)
        self.bn1 = nn.BatchNorm2d(24)  # Batch Normalization
        self.conv2 = nn.Conv2d(24, 36, kernel_size=5, stride=2, padding=2)
        self.bn2 = nn.BatchNorm2d(36)
        self.conv3 = nn.Conv2d(36, 48, kernel_size=5, stride=2, padding=2)
        self.bn3 = nn.BatchNorm2d(48)
        self.conv4 = nn.Conv2d(48, 64, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(64)
        self.conv5 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn5 = nn.BatchNorm2d(64)
        self.conv6 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn6 = nn.BatchNorm2d(128)

        # Fully connected layers
        self.fc1 = nn.Linear(28800, 1024)  # Flattened from convolutional output
        #self.fc1 = nn.Linear(128 * 6 * 6, 1024)  # Flattened from convolutional output
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 256)
        self.fc4 = nn.Linear(256, 3)  # Output steering angle

        # Dropout layers to prevent overfitting
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        # Apply convolutional layers with ReLU and Batch Normalization
        x = torch.relu(self.bn1(self.conv1(x)))
        x = torch.relu(self.bn2(self.conv2(x)))
        x = torch.relu(self.bn3(self.conv3(x)))
        x = torch.relu(self.bn4(self.conv4(x)))
        x = torch.relu(self.bn5(self.conv5(x)))
        x = torch.relu(self.bn6(self.conv6(x)))

        # Flatten the tensor for fully connected layers
        x = x.reshape(x.size(0), -1)

        # Apply fully connected layers with ReLU and Dropout
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout(x)
        x = torch.relu(self.fc3(x))

        # Final output layer for steering angle prediction
        x = self.fc4(x)

        return x


    '''