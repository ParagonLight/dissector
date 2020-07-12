import torch.nn as nn
import torch.nn.functional as F

class VGG16(nn.Module):
    def __init__(self, num_classes=1000):
        super(VGG16, self).__init__()
        self.conv1_1 = self.conv(3, 64)
        self.conv1_2 = self.conv(64, 64)
        self.conv2_1 = self.conv(64, 128)
        self.conv2_2 = self.conv(128, 128)
        self.conv3_1 = self.conv(128, 256)
        self.conv3_2 = self.conv(256, 256)
        self.conv3_3 = self.conv(256, 256)
        self.conv4_1 = self.conv(256, 512)
        self.conv4_2 = self.conv(512, 512)
        self.conv4_3 = self.conv(512, 512)
        self.conv5_1 = self.conv(512, 512)
        self.conv5_2 = self.conv(512, 512)
        self.conv5_3 = self.conv(512, 512)
        self.out = nn.Linear(512, num_classes)



    def conv(self, in_c, out_c):
        return nn.Conv2d(in_c, out_c, kernel_size = 3, padding = 1)
    def mp2(self, x):
        return F.max_pool2d(x, (2,2))

    def relu(self, x):
        return F.relu(x)

    def add(self, em, key, value):
        em[key] = value


    def forward(self, x):
        embeddings = {}

        x = self.relu(self.conv1_1(x))
        #self.add(embeddings, 'conv1_1', x)
        x = self.relu(self.conv1_2(x))
        #self.add(embeddings, 'conv1_2', x)
        x = self.mp2(x)
        self.add(embeddings, 'mp1', x)

        x = self.relu(self.conv2_1(x))
        #self.add(embeddings, 'conv2_1', x)
        x = self.relu(self.conv2_2(x))
        #self.add(embeddings, 'conv2_2', x)
        x = self.mp2(x)
        self.add(embeddings, 'mp2', x)

        x = self.relu(self.conv3_1(x))
        #self.add(embeddings, 'conv3_1', x)
        x = self.relu(self.conv3_2(x))
        #self.add(embeddings, 'conv3_2', x)
        x = self.relu(self.conv3_3(x))
        #self.add(embeddings, 'conv3_3', x)
        x = self.mp2(x)
        self.add(embeddings, 'mp3', x)

        x = self.relu(self.conv4_1(x))
        #self.add(embeddings, 'conv4_1', x)
        x = self.relu(self.conv4_2(x))
        #self.add(embeddings, 'conv4_2', x)
        x = self.relu(self.conv4_3(x))
        #self.add(embeddings, 'conv4_3', x)
        x = self.mp2(x)
        self.add(embeddings, 'mp4', x)

        x = self.relu(self.conv5_1(x))
        #self.add(embeddings, 'conv5_1', x)
        x = self.relu(self.conv5_2(x))
        #self.add(embeddings, 'conv5_2', x)
        x = self.relu(self.conv5_3(x))
        #self.add(embeddings, 'conv5_3', x)
        x = self.mp2(x)
        self.add(embeddings, 'mp5', x)
        x = x.view(x.size(0), -1)
        x = self.out(x)
        self.add(embeddings, 'out', x)
        return F.log_softmax(x, dim=1), embeddings
