import torch
import torch.nn as nn
import torch.nn.functional as F

class CustomCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(CustomCNN, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1)  # assuming RGB input
        self.activation1 = nn.ReLU()
        self.batchnorm1 = nn.BatchNorm2d(64)

        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.activation2 = nn.ReLU()
        self.batchnorm2 = nn.BatchNorm2d(64)

        self.pooling1 = nn.MaxPool2d(kernel_size=2)
        self.dropout1 = nn.Dropout(p=0.25)

        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.activation3 = nn.ReLU()
        self.batchnorm3 = nn.BatchNorm2d(128)

        self.conv4 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1)
        self.activation4 = nn.ReLU()
        self.batchnorm4 = nn.BatchNorm2d(128)

        self.pooling2 = nn.MaxPool2d(kernel_size=2)
        self.dropout2 = nn.Dropout(p=0.25)

        self.flatten = nn.Flatten()

        # Output size after two pooling layers (assuming input is 3x32x32 like CIFAR-10):
        # 32 -> 16 -> 8, so final feature map size is 128 x 8 x 8 = 8192
        self.fc1 = nn.Linear(128 * 8 * 8, 512)
        self.activation5 = nn.ReLU()
        self.dropout3 = nn.Dropout(p=0.25)

        self.fc3 = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.activation1(x)
        x = self.batchnorm1(x)

        x = self.conv2(x)
        x = self.activation2(x)
        x = self.batchnorm2(x)

        x = self.pooling1(x)
        x = self.dropout1(x)

        x = self.conv3(x)
        x = self.activation3(x)
        x = self.batchnorm3(x)

        x = self.conv4(x)
        x = self.activation4(x)
        x = self.batchnorm4(x)

        x = self.pooling2(x)
        x = self.dropout2(x)

        x = self.flatten(x)

        x = self.fc1(x)
        x = self.activation5(x)
        x = self.dropout3(x)

        x = self.fc3(x)
        return x
        # return F.log_softmax(x, dim=1)  # Using log_softmax for stability (e.g., with NLLLoss)

