import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
from tqdm import tqdm
import numpy as np
from PIL import Image
from sklearn.metrics import precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import models, transforms
from torch.autograd import Variable
from torch.nn import functional as F
import numpy as np
import cv2
import json

from tools.ai.optim_utils import *

import random
import torch
import numpy as np





#####################################################################
train_dir = 'crack_data_Pinggu/train/traindata' 
save_model_path = 'Pinggu model/1111result/1111CAM_model.pth'
batch_size = 8
batch_size_eval = 1
learning_rate = 0.0001
num_epochs = 200
######################################################################

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



# def set_seed(seed):
#     random.seed(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed(seed)
#     torch.cuda.manual_seed_all(seed)
#
#     torch.backends.cudnn.deterministic = True
#     torch.backends.cudnn.benchmark = False
#
# set_seed(2025)



class dilateattention(nn.Module):
    def __init__(self, in_channel, depth):
        super(dilateattention, self).__init__()
        self.batch_norm = nn.BatchNorm3d(in_channel)
        self.relu = nn.ReLU()
        self.atrous_block1 = nn.Conv3d(in_channel, depth, 1, 1)
        self.atrous_block2 = nn.Conv3d(in_channel, depth, 3, 1, padding=2, dilation=2, groups=in_channel)
        self.atrous_block3 = nn.Conv3d(in_channel, depth, 3, 1, padding=4, dilation=4, groups=in_channel)
        self.conv_1x1_output = nn.Conv3d(depth * 5, depth, 1, 1)
        self.softmax = nn.Softmax()


    def forward(self, x):
        v = self.atrous_block1(x)
        q = self.atrous_block2(x)
        k = self.atrous_block3(x)
        temp = k * q
        # dk = torch.std(temp)
        output = v * self.softmax(temp)
        output += self.relu(self.batch_norm(x))
        return output

class mymodel(nn.Module):
    def __init__(self, in_channel,classnum=2):
        super(mymodel, self).__init__()
        self.dim = 8
        self.conv3d1 = nn.Conv3d(in_channel, self.dim, kernel_size=(7,7,7), padding=3,stride=1)
        self.maxpooling1 = nn.MaxPool3d(kernel_size=2,stride=2)
        self.dialatiaon1 = dilateattention(self.dim,self.dim)
        self.conv3d2 = nn.Conv3d(self.dim, self.dim, kernel_size=(7, 7, 7), padding=3, stride=1)
        self.maxpooling2 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.dialatiaon2 = dilateattention(self.dim, self.dim)
        self.conv3d3 = nn.Conv3d(self.dim, self.dim, kernel_size=(7, 7, 7), padding=3, stride=1)
        self.maxpooling3 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.dialatiaon3 = dilateattention(self.dim, self.dim)

        self.conv1 = nn.Conv3d(1, 24, kernel_size=(7, 1, 1), stride=(2, 1, 1), bias=True)
        self.bn1 = nn.BatchNorm3d(24)
        self.activation1 = nn.ReLU()

        self.conv2 = nn.Conv3d(24, 24, kernel_size=(7, 1, 1), stride=1, padding=(3, 0, 0), padding_mode='replicate',
                               bias=True)
        self.bn2 = nn.BatchNorm3d(24)
        self.activation2 = nn.ReLU()
        self.conv3 = nn.Conv3d(24, 24, kernel_size=(7, 1, 1), stride=1, padding=(3, 0, 0), padding_mode='replicate',
                               bias=True)
        self.bn3 = nn.BatchNorm3d(24)
        self.activation3 = nn.ReLU()
     
        self.conv4 = nn.Conv3d(24, 2, kernel_size=(8, 1, 1), bias=True)
        self.bn4 = nn.BatchNorm3d(2)
        self.activation4 = nn.ReLU()

        self.conv5 = nn.Conv3d(1, 24, (7, 1, 1))
        self.bn5 = nn.BatchNorm3d(24)
        self.activation5 = nn.ReLU()

        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.transpose1 = nn.ConvTranspose3d(self.dim,self.dim,kernel_size=2,stride=2)
        self.conv2d1 = nn.Conv3d(self.dim,self.dim, kernel_size=(7,7,7),padding=(3,3,3),stride=1)
        self.transpose2 = nn.ConvTranspose3d(self.dim * 2, self.dim, kernel_size=2, stride=2)
        self.conv2d2 = nn.Conv3d(self.dim, self.dim, kernel_size=7, padding=3, stride=1)
        self.transpose3 = nn.ConvTranspose3d(self.dim * 2, self.dim, kernel_size=2, stride=2)
        self.conv2d3 = nn.Conv3d(self.dim, 1, kernel_size=7, padding=3, stride=1)
        self.final = nn.Conv2d(22, classnum, 3, 1, 1)
        self.final_cam = nn.Conv2d(2, classnum, 3, 1, 1)
        self.softmax = nn.Softmax()
        self.fc = nn.Linear(22, 2)  

    def forward(self,x):
        #spatial
        x1 = self.conv3d1(x)
        x1 = self.maxpooling1(x1)
        x1 = self.dialatiaon1(x1)
        x1 = self.conv3d2(x1)
        x1 = self.maxpooling2(x1)
        x1 = self.dialatiaon2(x1)
        x1 = self.conv3d3(x1)
        x1 = self.maxpooling3(x1)
        x1 = self.dialatiaon3(x1)
        x1 = self.conv2d3(x1)
        x1 = torch.squeeze(x1, dim=1)
        x1 = self.final(x1)

        cam_features = torch.squeeze(x1, dim=1)
        cam_features = self.final_cam(cam_features)
        x_gap = self.gap(cam_features)  
        x_gap = x_gap.squeeze()
        output = self.softmax(x_gap)
        

        return output

class HyperspectralDataset(Dataset):
    def __init__(self, img_dir,transform=None):
        self.img_dir = img_dir
        self.transform = transform
        self.classes, self.class_to_idx = self._find_classes(img_dir)
        self.samples = self._make_dataset(img_dir, self.class_to_idx)

    def _find_classes(self, dir):
        classes = [d.name for d in os.scandir(dir) if d.is_dir()]
        classes.sort()
        class_to_idx = {cls_name: idx for idx, cls_name in enumerate(classes)}
        return classes, class_to_idx

    def _make_dataset(self, dir, class_to_idx):
        images = []
        for target_class in self.classes:
            class_idx = class_to_idx[target_class]
            target_dir = os.path.join(dir, target_class)
            if not os.path.isdir(target_dir):
                continue
            for root, _, fnames in sorted(os.walk(target_dir)):
                for fname in sorted(fnames):
                    if fname.lower().endswith(('.npy',)):
                        path = os.path.join(root, fname)
                        item = (path, class_idx) 
                        images.append(item)
        return images

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, target = self.samples[idx]
        img = np.load(img_path)

        img = img.astype(np.float32)
        img = img[:176, :, :]
        img = torch.from_numpy(img).unsqueeze(0)  # (1, 176, 224, 224)
        num_classes = len(self.classes)
        one_hot_target = torch.zeros(num_classes)
        one_hot_target[target] = 1

        return img,one_hot_target




train_dataset = HyperspectralDataset(train_dir)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)


model = mymodel(in_channel=1)
model.to(device)

val_iteration = len(train_loader)
max_iteration = num_epochs * val_iteration
criterion = nn.MultiLabelSoftMarginLoss(reduction='none').cuda()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)


train_losses = [] 


for epoch in range(num_epochs):
    model.train()  
    epoch_loss = 0.0  
    for img, target in tqdm(train_loader):
        img, target = img.to(device), target.to(device)

     
        output = model(img)
        output = output.unsqueeze(0) 

        loss = criterion(output, target).mean()
        epoch_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    average_loss = epoch_loss / len(train_loader)
    train_losses.append(average_loss)
    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {average_loss:.4f}')
    torch.save(model.state_dict(), save_model_path)

    



