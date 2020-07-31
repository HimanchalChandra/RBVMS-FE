import torch
import torch.nn as nn
import torch.nn.functional as F

class Recog_Net(nn.Module):
    
    def __init__(self):

        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 7, stride = 2)
        self.conv2a = nn.Conv2d(64, 64, 1, stride = 1)
        self.conv2 = nn.Conv2d(64, 192, 3, stride = 1)
        self.conv3a = nn.Conv2d(192, 192, 1, stride = 1)
        self.conv3 = nn.Conv2d(192, 384, 3, stride = 1)
        self.conv4a = nn.Conv2d(384, 384, 1, stride = 1)
        self.conv4 = nn.Conv2d(384, 256, 3, stride = 1)
        self.conv5a = nn.Conv2d(256, 256, 1, stride = 1)
        self.conv5 = nn.Conv2d(256, 256, 3, stride = 1)
        self.conv6a = nn.Conv2d(256, 256, 1, stride = 1)
        self.conv6 = nn.Conv2d(256, 256, 3, stride = 1)
        self.fc1 = nn.Linear(256 * 5 * 5, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, 128)
        

    def forward(self, x):
        
        x = F.max_pool2d(F.relu(self.conv1(x)), (3,3), stride = 2)
        x = F.relu(self.conv2a(x))
        x = F.max_pool2d(F.relu(self.conv2(x)), (3,3), stride = 2)
        x = F.relu(self.conv3a(x))
        x = F.max_pool2d(F.relu(self.conv3(x)), (3,3), stride = 2)
        x = F.relu(self.conv4a(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5a(x))
        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6a(x))
        x = F.relu(self.conv6(x))
        x = x.view(-1, 256 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        x = F.normalize(x, p=2, dim=1) 
        x = x * 10

        return x