from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import os
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score
import random

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


#######################################################################################
data_dir = 'crack_data_Leeds/train/traindata/0' 
label_dir = 'Leeds model/1111result/1111pse-label'
model_path = 'Leeds model/1111result/1111segmentation_model.pth'
batch_size = 8 
batch_size_test = 8
learning_rate = 0.0001
num_epochs = 50
######################################################################################


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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
    def __init__(self, in_channel,classnum=1):
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
        self.transpose1 = nn.ConvTranspose3d(self.dim,self.dim,kernel_size=2,stride=2)
        self.conv2d1 = nn.Conv3d(self.dim,self.dim, kernel_size=(7,7,7),padding=(3,3,3),stride=1)
        self.transpose2 = nn.ConvTranspose3d(self.dim * 2, self.dim, kernel_size=2, stride=2)
        self.conv2d2 = nn.Conv3d(self.dim, self.dim, kernel_size=7, padding=3, stride=1)
        self.transpose3 = nn.ConvTranspose3d(self.dim * 2, self.dim, kernel_size=2, stride=2)
        self.conv2d3 = nn.Conv3d(self.dim, 1, kernel_size=7, padding=3, stride=1)
        self.final = nn.Conv2d(136, classnum, 3, 1, 1)

    def forward(self,x):
        x1 = self.conv3d1(x)
        x1 = self.maxpooling1(x1)
        x1 = self.dialatiaon1(x1)
        x2 = self.conv3d2(x1)
        x2 = self.maxpooling2(x2)
        x2 = self.dialatiaon2(x2)
        x3 = self.conv3d3(x2)
        x3 = self.maxpooling3(x3)
        x3 = self.dialatiaon3(x3)
        x4 = self.transpose1(x3)
        x4 = self.conv2d1(x4)
        x4 = torch.cat([x4, x2], dim=1)
        x5 = self.transpose2(x4)
        x5 = self.conv2d2(x5)
        x6 = torch.cat([x5,x1], dim=1)
        x6 = self.transpose3(x6)
        x6 = self.conv2d3(x6)
        x6 = torch.squeeze(x6,dim=1)
        x6 = self.final(x6)
        return x6



class HyperspectralDataset(Dataset):
    def __init__(self, img_dir, label_dir):
        self.img_dir = img_dir
        self.label_dir = label_dir

        self.img_files = {os.path.splitext(f)[0]: f for f in os.listdir(img_dir) if f.endswith('.npy')}

        self.label_files = sorted([f for f in os.listdir(label_dir)])

        assert len(self.img_files) >= len(self.label_files), "There must be at least as many images as labels."

        for label_file in self.label_files:
            _, ext = os.path.splitext(label_file)
            assert ext.lower() in ['.png', '.npy'], f"Unsupported label file format: {ext}"

    def __len__(self):
       
        return len(self.label_files)

    def __getitem__(self, idx):
      
        label_file = self.label_files[idx]
        label_path = os.path.join(self.label_dir, label_file)
        
       
        label_key = os.path.splitext(label_file)[0]
        
      
        if label_key in self.img_files:
            img_file = self.img_files[label_key]
        else:
           
            matched = False
            for key in self.img_files.keys():
                if key.startswith(label_key):
                    img_file = self.img_files[key]
                    matched = True
                    break
            if not matched:
                raise FileNotFoundError(f"No matching image found for label: {label_file}")
        
        img_path = os.path.join(self.img_dir, img_file)

        img = np.load(img_path)

        _, ext = os.path.splitext(label_path)
        if ext.lower() == '.png':
          
            label = Image.open(label_path)
            label = np.array(label, dtype=np.float32) / 255.0
            # label = 1 - label
        elif ext.lower() == '.npy':
          
            label = np.load(label_path)
            label = label.astype(np.float32)
        else:
           
            raise ValueError(f"Unexpected label file format: {ext}")
     
        img = img.astype(np.float32)  
        img = img.transpose(2, 0, 1) 
        img = img[:136, :, :]

        img_tensor = torch.from_numpy(img).unsqueeze(0)  
        label_tensor = torch.from_numpy(label).unsqueeze(0)  

        return img_tensor, label_tensor

def compute_iou(pred, true, classes, ignore_index=255):
    iou_dict = {}
    total_iou = 0
    total_classes = len(classes)

    for cls in classes:
        pred_cls = (pred == cls)
        true_cls = (true == cls)

        if ignore_index is not None:
            ignore_mask = (true != ignore_index)
            pred_cls = pred_cls & ignore_mask
            true_cls = true_cls & ignore_mask

        intersection = np.logical_and(pred_cls, true_cls).sum()
        union = np.logical_or(pred_cls, true_cls).sum()

        if union == 0:
            iou = float('nan')  
        else:
            iou = intersection / union

        iou_dict[cls] = iou
        total_iou += iou

    mean_iou = total_iou / total_classes if total_classes > 0 else float('nan')

    return {'iou_per_class': iou_dict, 'mean_iou': mean_iou}



dataset = HyperspectralDataset(data_dir, label_dir)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)


model = mymodel(in_channel=1)
model.to(device)


criterion = nn.BCEWithLogitsLoss() 
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
train_losses = [] 

# 训练循环
for epoch in range(num_epochs):
    model.train() 
    epoch_loss = 0.0  
    for img, label in tqdm(dataloader):
        img, label = img.to(device), label.to(device)

        output = model(img)

        loss = criterion(output, label)
        epoch_loss += loss.item() 

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    avg_loss = epoch_loss / len(dataloader)
    train_losses.append(avg_loss)

    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}')
    torch.save(model.state_dict(), model_path)


       
   





