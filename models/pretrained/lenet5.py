import torch
import torch.nn as nn
import torch.nn.functional as F

class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5, padding=2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        #self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(16*5*5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        self.lenet5_layers = ['conv1', 'conv2', 'fc1', 'fc2']

    def add(self, em, key, value):
        em[key] = value

    def forward(self, x):
        embeddings = {}
        x = F.max_pool2d(F.relu(self.conv1(x)), (2,2))
        self.add(embeddings, 'conv1', x)
        x = F.max_pool2d(F.relu(self.conv2(x)), (2,2))
        self.add(embeddings, 'conv2', x)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        self.add(embeddings, 'fc1', x)
        x = F.relu(self.fc2(x))
        self.add(embeddings, 'fc2', x)
        x = self.fc3(x)
        self.add(embeddings, 'out', x)
        self.add(embeddings, 'fc3', x)
        return F.log_softmax(x, dim=1), embeddings

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


