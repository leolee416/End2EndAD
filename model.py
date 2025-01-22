import torch
import torch.nn as nn
import torch.nn.functional as F

class ComplexCNN(nn.Module):
    def __init__(self):
        super(ComplexCNN, self).__init__()

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

        # Fully connected layers for the image input
        self.fc1 = nn.Linear(512 * 2 * 4, 1024)
        self.fc2 = nn.Linear(1024, 512) 
        self.fc3 = nn.Linear(512, 100)

        # Define fully connected layers for velocity input
        self.fc_speed = nn.Linear(1, 20) 

        # Combine image features and velocity features
        self.fc_combined = nn.Linear(100 + 20, 512)

        self.dropout = nn.Dropout(0.5)

        # Outputs
        self.fc_end = nn.Linear(512, 3)


    def forward(self, image_input, speed_input):
        # Process image input through convolutional layers
        x = F.relu(self.bn1(self.conv1(image_input)))  # Apply BatchNorm
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))
        x = F.relu(self.bn6(self.conv6(x)))  # Last convolutional layer

        x = x.reshape(x.size(0), -1)  # Flatten the output
        x = F.relu(self.fc1(x))
        x = self.dropout(x)  
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = F.relu(self.fc3(x))
        x = self.dropout(x)  # Output from image feature extraction

        # Process velocity input
        speed_input = speed_input.reshape(-1, 1)
        v = F.relu(self.fc_speed(speed_input))  # Apply fully connected layer on speed

        # Concatenate the image and velocity features
        combined = torch.cat((x, v), dim=1)  # Concatenate along the feature dimension

        # Combine features through a fully connected layer
        combined = F.relu(self.fc_combined(combined))
        

        result = self.fc_end(combined)

        # Compute the outputs (throttle, brake, steering)
        # throttle_output = torch.sigmoid(self.fc_throttle(combined))
        # brake_output = torch.sigmoid(self.fc_brake(combined))
        # steering_output = torch.tanh(self.fc_steering(combined))

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