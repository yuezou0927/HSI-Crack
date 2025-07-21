import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
from tqdm import tqdm
from PIL import Image
import matplotlib.pyplot as plt

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#######################################################################
data_dir = 'crack_data_Pinggu/test/image0'  
label_dir = 'crack_data_Pinggu/test/label0'  
prediction_path = 'Pinggu model/1111result/1111prediction'
model_path = 'Pinggu model/1111result/1111segmentation.pth'
batch_size = 8  
##########################################################################


def calculate_region_metrics(pred_masks, true_masks, num_classes=2):
    """基于区域（完整掩码）计算IoU、精确率、召回率和F1分数"""
    total_tp = np.zeros(num_classes) 
    total_fp = np.zeros(num_classes) 
    total_fn = np.zeros(num_classes) 
    
    for pred, true in zip(pred_masks, true_masks):
        for c in range(num_classes):
           
            pred_c = (pred == c).astype(np.uint8)
            true_c = (true == c).astype(np.uint8)
            
            
            total_tp[c] += np.logical_and(pred_c, true_c).sum()
            total_fp[c] += np.logical_and(pred_c, np.logical_not(true_c)).sum()
            total_fn[c] += np.logical_and(np.logical_not(pred_c), true_c).sum()
    
   
    class_precision = []
    class_recall = []
    class_f1 = []
    class_iou = []
    
    for c in range(num_classes):
        # 精确率 = TP / (TP + FP)
        precision = total_tp[c] / (total_tp[c] + total_fp[c]) if (total_tp[c] + total_fp[c]) > 0 else 0
        # 召回率 = TP / (TP + FN)
        recall = total_tp[c] / (total_tp[c] + total_fn[c]) if (total_tp[c] + total_fn[c]) > 0 else 0
        # F1分数 = 2 * (P*R) / (P+R)
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        # IoU = TP / (TP + FP + FN)
        iou = total_tp[c] / (total_tp[c] + total_fp[c] + total_fn[c]) if (total_tp[c] + total_fp[c] + total_fn[c]) > 0 else 0
        
        class_precision.append(precision)
        class_recall.append(recall)
        class_f1.append(f1)
        class_iou.append(iou)
    
    # 计算平均mIoU
    miou = np.mean(class_iou)
    return class_precision, class_recall, class_f1, class_iou, miou


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

    def register_feature_hook(self, layer_name):
        def hook_fn(module, input, output):
            self.feature_maps = output.detach()
        layer = getattr(self, layer_name)
        self.feature_hook = layer.register_forward_hook(hook_fn)

    def forward(self, x):
        v = self.atrous_block1(x)
        q = self.atrous_block2(x)
        k = self.atrous_block3(x)
        temp = k * q
        output = v * self.softmax(temp)
        output += self.relu(self.batch_norm(x))
        return output


class mymodel(nn.Module):
    def __init__(self, in_channel, classnum=1):
        super(mymodel, self).__init__()
        self.dim = 8
        self.conv3d1 = nn.Conv3d(in_channel, self.dim, kernel_size=(7, 7, 7), padding=3, stride=1)
        self.maxpooling1 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.dialatiaon1 = dilateattention(self.dim, self.dim)
        self.conv3d2 = nn.Conv3d(self.dim, self.dim, kernel_size=(7, 7, 7), padding=3, stride=1)
        self.maxpooling2 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.dialatiaon2 = dilateattention(self.dim, self.dim)
        self.conv3d3 = nn.Conv3d(self.dim, self.dim, kernel_size=(7, 7, 7), padding=3, stride=1)
        self.maxpooling3 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.dialatiaon3 = dilateattention(self.dim, self.dim)
        self.transpose1 = nn.ConvTranspose3d(self.dim, self.dim, kernel_size=2, stride=2)
        self.conv2d1 = nn.Conv3d(self.dim, self.dim, kernel_size=(7, 7, 7), padding=(3, 3, 3), stride=1)
        self.transpose2 = nn.ConvTranspose3d(self.dim * 2, self.dim, kernel_size=2, stride=2)
        self.conv2d2 = nn.Conv3d(self.dim, self.dim, kernel_size=7, padding=3, stride=1)
        self.transpose3 = nn.ConvTranspose3d(self.dim * 2, self.dim, kernel_size=2, stride=2)
        self.conv2d3 = nn.Conv3d(self.dim, 1, kernel_size=7, padding=3, stride=1)
        self.final = nn.Conv2d(176, classnum, 3, 1, 1)

    def forward(self, x):
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
        x6 = torch.cat([x5, x1], dim=1)
        x6 = self.transpose3(x6)
        x6 = self.conv2d3(x6)
        x6 = torch.squeeze(x6, dim=1)
        x6 = self.final(x6)
        return x6


class HyperspectralDataset(Dataset):
    def __init__(self, img_dir, label_dir):
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.img_files = sorted([f for f in os.listdir(img_dir) if f.endswith('.npy')])
        self.label_files = sorted([f for f in os.listdir(label_dir)])
        assert len(self.img_files) == len(self.label_files), "图像和标签数量不匹配"
        for label_file in self.label_files:
            _, ext = os.path.splitext(label_file)
            assert ext.lower() in ['.png', '.npy'], f"不支持的标签格式: {ext}"

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_files[idx])
        label_path = os.path.join(self.label_dir, self.label_files[idx])
        img = np.load(img_path)
        _, ext = os.path.splitext(label_path)
        if ext.lower() == '.png':
            label = Image.open(label_path)
            label = np.array(label, dtype=np.float32) / 255.0  
        elif ext.lower() == '.npy':
            label = np.load(label_path)
            label = label.astype(np.float32)
        else:
            raise ValueError(f"不支持的标签格式: {ext}")
        img = img.astype(np.float32)
        img = img[:176, :, :]  
        img_tensor = torch.from_numpy(img).unsqueeze(0)  
        label_tensor = torch.from_numpy(label).unsqueeze(0)  
        file_name = os.path.splitext(self.img_files[idx])[0]
        return img_tensor, label_tensor, file_name



dataset = HyperspectralDataset(data_dir, label_dir)
testloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)


model = mymodel(in_channel=1)
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()  


if not os.path.exists(prediction_path):
    os.makedirs(prediction_path)


criterion = nn.BCEWithLogitsLoss()

test_loss = 0.0
pred_masks_list = []  
true_masks_list = []  



with torch.no_grad():
    for img, label, file_names in testloader:
        img, label = img.to(device), label.to(device)
        output = model(img)  
        loss = criterion(output, label)  
        test_loss += loss.item() 
        
        
        probabilities = torch.sigmoid(output)
       
        predictions = (probabilities.cpu().numpy() > 0.003).astype(int)
        labels = label.cpu().numpy().astype(int)  

       
        for i in range(len(predictions)):
           
            pred_img = predictions[i].squeeze() 
            save_path = os.path.join(prediction_path, f'{file_names[i]}.png')
            plt.imsave(save_path, pred_img, cmap='gray')  

            pred_masks_list.append(pred_img)  
            true_masks_list.append(labels[i].squeeze())  



test_loss /= len(testloader)
print(f"测试集平均损失: {test_loss:.4f}")


class_precision, class_recall, class_f1, class_iou, miou = calculate_region_metrics(
    pred_masks_list, true_masks_list, num_classes=2
)

# 输出结果
print("\n===== 基于区域的评估指标 =====")
for class_id in range(2):
    print(f"类别 {class_id}:")
    print(f"  精确率 (Precision): {class_precision[class_id]:.4f}")
    print(f"  召回率 (Recall): {class_recall[class_id]:.4f}")
    print(f"  F1分数 (F1-Score): {class_f1[class_id]:.4f}")
    print(f"  IoU: {class_iou[class_id]:.4f}")
print(f"mIoU: {miou:.4f}")