import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
from tqdm import tqdm
from PIL import Image
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


###########################################################################
data_dir = 'crack_data_Leeds/test/0'  
label_dir = 'crack_data_Leeds/test/label0'  
prediction_path = 'Leeds model/1111result/1111prediction'
model_path = 'Leeds model/1111result/1111segmentation_model.pth'
batch_size = 8 
#############################################################################

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 自定义评估指标函数
def custom_precision(y_pred, y_target):
    true_positives = np.sum((y_pred == 0) & (y_target == 0))
    false_positives = np.sum((y_pred == 0) & (y_target != 0))
    if true_positives + false_positives == 0:
        return 0
    return true_positives / (true_positives + false_positives)

def custom_accuracy(y_pred, y_target):
    # 只考虑 0 类
    zero_class_indices = np.where(y_target == 0)[0]
    y_pred_zero = y_pred[zero_class_indices]
    y_target_zero = y_target[zero_class_indices]
    correct_predictions = np.sum(y_pred_zero == y_target_zero)
    total_pixels = y_pred_zero.size
    if total_pixels == 0:
        return 0
    return correct_predictions / total_pixels

def custom_f1_score(y_pred, y_target):
    precision = custom_precision(y_pred, y_target)
    recall = custom_recall(y_pred, y_target)
    if precision + recall == 0:
        return 0
    return 2 * (precision * recall) / (precision + recall)

def custom_recall(y_pred, y_target):
    true_positives = np.sum((y_pred == 0) & (y_target == 0))
    false_negatives = np.sum((y_pred != 0) & (y_target == 0))
    if true_positives + false_negatives == 0:
        return 0
    return true_positives / (true_positives + false_negatives)

def custom_MIoU(y_pred, y_target):
    intersection = np.sum((y_pred == 0) & (y_target == 0))
    union = np.sum((y_pred == 0) | (y_target == 0))
    if union == 0:
        return 0
    return intersection / union

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
        self.final = nn.Conv2d(136, classnum, 3, 1, 1)

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
        assert len(self.img_files) == len(
            self.label_files), "The number of image files must match the number of label files."
        for label_file in self.label_files:
            _, ext = os.path.splitext(label_file)
            assert ext.lower() in ['.png', '.npy'], f"Unsupported label file format: {ext}"

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
            label = 1 - label
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

# 实例化数据集
dataset = HyperspectralDataset(data_dir, label_dir)
testloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

model = mymodel(in_channel=1)
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)  
model.eval()


criterion = nn.BCEWithLogitsLoss()


test_loss = 0.0
all_labels = []
all_predictions = []
all_cm = np.zeros((2, 2))
num_samples = 0

if not os.path.exists(prediction_path):
    os.makedirs(prediction_path)

with torch.no_grad():
    image_index = 0
    for img, label in testloader:
        img, label = img.to(device), label.to(device)

        output = model(img)

        loss = criterion(output, label)
        test_loss += loss.item()
        probabilities = torch.sigmoid(output)

        predictions = (probabilities.cpu().numpy() > 0.3).astype(int)  
        labels = label.cpu().numpy().astype(int)

        for i in range(predictions.shape[0]):
            pred = predictions[i].flatten()
            true = labels[i].flatten()

            cm = confusion_matrix(true, pred)
            all_cm += cm
            num_samples += 1

            pred_img = pred.reshape(200, 200)
            save_path = os.path.join(prediction_path, f'prediction_{image_index}.png')
            plt.imsave(save_path, pred_img, cmap='gray')
            image_index += 1

        predictions = predictions.flatten()
        labels = labels.flatten()

        all_labels.extend(labels)
        all_predictions.extend(predictions)

test_loss /= len(testloader)


print("Confusion Matrix:")
print(all_cm)

TP_0 = all_cm[0, 0]  # 类别0的真正例
FP_0 = all_cm[1, 0]  # 类别0的假正例
FN_0 = all_cm[0, 1]  # 类别0的假负例
TN_0 = all_cm[1, 1]  # 类别0的真负例

TP_1 = all_cm[1, 1]  # 类别1的真正例
FP_1 = all_cm[0, 1]  # 类别1的假正例
FN_1 = all_cm[1, 0]  # 类别1的假负例
TN_1 = all_cm[0, 0]  # 类别1的真负例

  
precision_0 = TP_0 / (TP_0 + FP_0) if (TP_0 + FP_0) > 0 else 0
recall_0 = TP_0 / (TP_0 + FN_0) if (TP_0 + FN_0) > 0 else 0
f1_0 = 2 * (precision_0 * recall_0) / (precision_0 + recall_0) if (precision_0 + recall_0) > 0 else 0

precision_1 = TP_1 / (TP_1 + FP_1) if (TP_1 + FP_1) > 0 else 0
recall_1 = TP_1 / (TP_1 + FN_1) if (TP_1 + FN_1) > 0 else 0
f1_1 = 2 * (precision_1 * recall_1) / (precision_1 + recall_1) if (precision_1 + recall_1) > 0 else 0

    
iou_0 = TP_0 / (TP_0 + FP_0 + FN_0) if (TP_0 + FP_0 + FN_0) > 0 else 0
iou_1 = TP_1 / (TP_1 + FP_1 + FN_1) if (TP_1 + FP_1 + FN_1) > 0 else 0

    
mIoU = (iou_0 + iou_1) / 2

  
print("类别 0 的精确度：", precision_0)
print("类别 0 的召回率：", recall_0)
print("类别 0 的 F1 分数：", f1_0)
print("类别 0 的 IoU：", iou_0)

print("类别 1 的精确度：", precision_1)
print("类别 1 的召回率：", recall_1)
print("类别 1 的 F1 分数：", f1_1)
print("类别 1 的 IoU：", iou_1)
print("平均 IoU：", mIoU)

