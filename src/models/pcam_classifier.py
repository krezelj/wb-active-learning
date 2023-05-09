import torch
import torch.nn as nn
import torch.nn.functional as F


class PCAMClassifier(nn.Module):
    """Implemented by paper: http://cs230.stanford.edu/projects_winter_2019/posters/15813053.pdf"""
    def __init__(self, n_input_channels=3, n_conv_output_channels=16, k=3, s=1, pad=1, p = 0.5):
        super().__init__()
        # 1. Convolutional layers
        # Single image is in shape: 3x96x96 (CxHxW, H==W), RGB images
        self.conv1 = nn.Conv2d(in_channels = n_input_channels, out_channels = n_conv_output_channels, kernel_size = k, stride = s, padding = pad)
        self.bn1 = nn.BatchNorm2d(n_conv_output_channels)
        self.conv2 = nn.Conv2d(in_channels = n_conv_output_channels, out_channels = 2*n_conv_output_channels, kernel_size = k, stride = s, padding = pad)
        self.bn2 = nn.BatchNorm2d(2*n_conv_output_channels)
        self.conv3 = nn.Conv2d(in_channels = 2*n_conv_output_channels, out_channels = 4*n_conv_output_channels, kernel_size = k, stride = s, padding = pad)
        self.bn3 = nn.BatchNorm2d(4*n_conv_output_channels)
        self.conv4 = nn.Conv2d(in_channels = 4*n_conv_output_channels, out_channels = 8*n_conv_output_channels, kernel_size = k, stride = s, padding = pad)
        self.bn4 = nn.BatchNorm2d(8*n_conv_output_channels)
        self.pool = nn.MaxPool2d(kernel_size = k - 1, stride = 2*s, padding = pad - pad)
        
        self.dropout = nn.Dropout(p = p)
        
        # 2. FC layers to final output
        self.fc1 = nn.Linear(in_features = 288*n_conv_output_channels, out_features = 32*n_conv_output_channels)
        self.fc_bn1 = nn.BatchNorm1d(32*n_conv_output_channels)
        self.fc2 = nn.Linear(in_features = 32*n_conv_output_channels, out_features = 16*n_conv_output_channels)
        self.fc_bn2 = nn.BatchNorm1d(16*n_conv_output_channels)
        self.fc3 = nn.Linear(in_features = 16*n_conv_output_channels, out_features = 8*n_conv_output_channels)
        self.fc_bn3 = nn.BatchNorm1d(8*n_conv_output_channels)
        self.fc4 = nn.Linear(in_features = 8*n_conv_output_channels, out_features = 2)

    def forward(self, x):
        # Convolution Layers, followed by Batch Normalizations, Maxpool, and ReLU
        x = self.bn1(self.conv1(x))                      # batch_size x 96 x 96 x 16
        x = self.pool(F.relu(x))                         # batch_size x 48 x 48 x 16
        x = self.bn2(self.conv2(x))                      # batch_size x 48 x 48 x 32
        x = self.pool(F.relu(x))                         # batch_size x 24 x 24 x 32
        x = self.bn3(self.conv3(x))                      # batch_size x 24 x 24 x 64
        x = self.pool(F.relu(x))                         # batch_size x 12 x 12 x 64
        x = self.bn4(self.conv4(x))                      # batch_size x 12 x 12 x 128
        x = self.pool(F.relu(x))                         # batch_size x  6 x  6 x 128
        # Flatten the output for each image
        x = x.reshape(-1, self.num_flat_features(x))        # batch_size x 6*6*128
        
        # Apply 4 FC Layers
        x = self.fc1(x)
        x = self.fc_bn1(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        x = self.fc2(x)
        x = self.fc_bn2(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        x = self.fc3(x)
        x = self.fc_bn3(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        x = self.fc4(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features
    

class PCAMClassifierSmall(nn.Module):
    """Implemented by paper: http://cs230.stanford.edu/projects_winter_2019/posters/15813053.pdf"""
    def __init__(self, n_input_channels=3, n_conv_output_channels=16, k=3, s=1, pad=1, p = 0.5):
        super().__init__()
        # 1. Convolutional layers
        # Single image is in shape: 3x96x96 (CxHxW, H==W), RGB images
        self.conv1 = nn.Conv2d(in_channels = n_input_channels, out_channels = n_conv_output_channels, kernel_size = k, stride = s, padding = pad)
        self.bn1 = nn.BatchNorm2d(n_conv_output_channels)
        self.conv2 = nn.Conv2d(in_channels = n_conv_output_channels, out_channels = 2*n_conv_output_channels, kernel_size = k, stride = s, padding = pad)
        self.bn2 = nn.BatchNorm2d(2*n_conv_output_channels)
        self.conv3 = nn.Conv2d(in_channels = 2*n_conv_output_channels, out_channels = 4*n_conv_output_channels, kernel_size = k, stride = s, padding = pad)
        self.bn3 = nn.BatchNorm2d(4*n_conv_output_channels)
        self.conv4 = nn.Conv2d(in_channels = 4*n_conv_output_channels, out_channels = 8*n_conv_output_channels, kernel_size = k, stride = s, padding = pad)
        self.bn4 = nn.BatchNorm2d(8*n_conv_output_channels)
        self.pool = nn.MaxPool2d(kernel_size = k - 1, stride = 2*s, padding = pad - pad)
        
        self.dropout = nn.Dropout(p = p)
        
        # 2. FC layers to final output
        self.fc1 = nn.Linear(in_features = 32*n_conv_output_channels, out_features = 32*n_conv_output_channels)
        self.fc_bn1 = nn.BatchNorm1d(32*n_conv_output_channels)
        self.fc2 = nn.Linear(in_features = 32*n_conv_output_channels, out_features = 16*n_conv_output_channels)
        self.fc_bn2 = nn.BatchNorm1d(16*n_conv_output_channels)
        self.fc3 = nn.Linear(in_features = 16*n_conv_output_channels, out_features = 8*n_conv_output_channels)
        self.fc_bn3 = nn.BatchNorm1d(8*n_conv_output_channels)
        self.fc4 = nn.Linear(in_features = 8*n_conv_output_channels, out_features = 2)

    def forward(self, x):
        # Convolution Layers, followed by Batch Normalizations, Maxpool, and ReLU
        x = self.bn1(self.conv1(x))                      # batch_size x 32 x 32 x 16
        x = self.pool(F.relu(x))                         # batch_size x 16 x 16 x 16
        x = self.bn2(self.conv2(x))                      # batch_size x 16 x 16 x 32
        x = self.pool(F.relu(x))                         # batch_size x 8 x 8 x 32
        x = self.bn3(self.conv3(x))                      # batch_size x 8 x 8 x 64
        x = self.pool(F.relu(x))                         # batch_size x 4 x 4 x 64
        x = self.bn4(self.conv4(x))                      # batch_size x 4 x 4 x 128
        x = self.pool(F.relu(x))                         # batch_size x  2 x  2 x 128
        # Flatten the output for each image
        x = x.reshape(-1, self.num_flat_features(x))        # batch_size x 2*2*128
        
        # Apply 4 FC Layers
        x = self.fc1(x)
        x = self.fc_bn1(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        x = self.fc2(x)
        x = self.fc_bn2(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        x = self.fc3(x)
        x = self.fc_bn3(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        x = self.fc4(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

# class PCAMClassifierSmall(nn.Module):
#     def __init__(self, n_input_channels=3, n_conv_output_channels=16, k=3, s=1, pad=1, p = 0.5):
#         super().__init__()
#         # 1. Convolutional layers
#         # Single image is in shape: 3x32x32 (CxHxW, H==W), RGB images
#         self.conv1 = nn.Conv2d(in_channels = n_input_channels, out_channels = n_conv_output_channels, kernel_size = k, stride = s, padding = pad)
#         self.bn1 = nn.BatchNorm2d(n_conv_output_channels)
#         self.conv2 = nn.Conv2d(in_channels = n_conv_output_channels, out_channels = 2*n_conv_output_channels, kernel_size = k, stride = s, padding = pad)
#         self.bn2 = nn.BatchNorm2d(2*n_conv_output_channels)
#         # self.conv3 = nn.Conv2d(in_channels = 2*n_conv_output_channels, out_channels = 4*n_conv_output_channels, kernel_size = k, stride = s, padding = pad)
#         # self.bn3 = nn.BatchNorm2d(4*n_conv_output_channels)
#         # self.conv4 = nn.Conv2d(in_channels = 4*n_conv_output_channels, out_channels = 8*n_conv_output_channels, kernel_size = k, stride = s, padding = pad)
#         # self.bn4 = nn.BatchNorm2d(8*n_conv_output_channels)
#         self.pool = nn.MaxPool2d(kernel_size = k - 1, stride = 2*s, padding = pad - pad)
        
#         self.dropout = nn.Dropout(p = p)
        
#         # 2. FC layers to final output
#         self.fc1 = nn.Linear(in_features = 128*n_conv_output_channels, out_features = 32*n_conv_output_channels)
#         self.fc_bn1 = nn.BatchNorm1d(32*n_conv_output_channels)
#         self.fc2 = nn.Linear(in_features = 32*n_conv_output_channels, out_features = 16*n_conv_output_channels)
#         self.fc_bn2 = nn.BatchNorm1d(16*n_conv_output_channels)
#         self.fc3 = nn.Linear(in_features = 16*n_conv_output_channels, out_features = 8*n_conv_output_channels)
#         self.fc_bn3 = nn.BatchNorm1d(8*n_conv_output_channels)
#         self.fc4 = nn.Linear(in_features = 8*n_conv_output_channels, out_features = 2)

#     def forward(self, x):
#         # Convolution Layers, followed by Batch Normalizations, Maxpool, and ReLU
#         x = self.bn1(self.conv1(x))                      # batch_size x 32 x 32 x 16
#         x = self.pool(F.relu(x))                         # batch_size x 16 x 16 x 16
#         x = self.bn2(self.conv2(x))                      # batch_size x 16 x 16 x 32
#         x = self.pool(F.relu(x))                         # batch_size x 8 x 8 x 32
#         # x = self.bn3(self.conv3(x))                      # batch_size x 8 x 8 x 64
#         # x = self.pool(F.relu(x))                         # batch_size x 12 x 12 x 64
#         # x = self.bn4(self.conv4(x))                      # batch_size x 12 x 12 x 128
#         # x = self.pool(F.relu(x))                         # batch_size x  6 x  6 x 128
#         # Flatten the output for each image
#         x = x.reshape(-1, self.num_flat_features(x))        # batch_size x 8*8*64
        
#         # Apply 4 FC Layers
#         x = self.fc1(x)
#         x = self.fc_bn1(x)
#         x = F.relu(x)
#         x = self.dropout(x)
        
#         x = self.fc2(x)
#         x = self.fc_bn2(x)
#         x = F.relu(x)
#         x = self.dropout(x)
        
#         x = self.fc3(x)
#         x = self.fc_bn3(x)
#         x = F.relu(x)
#         x = self.dropout(x)
        
#         x = self.fc4(x)
#         return x

#     def num_flat_features(self, x):
#         size = x.size()[1:]  # all dimensions except the batch dimension
#         num_features = 1
#         for s in size:
#             num_features *= s
#         return num_features