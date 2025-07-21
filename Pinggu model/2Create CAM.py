import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import os
from torchvision import models, transforms
from torch.autograd import Variable
import json

from tools.ai.optim_utils import *


###########################################################
LABELS_file = 'imagenet-simple-labels.json'
image_file_path = 'crack_data_Pinggu/train/traindata/0'
CAM_output_folder = 'Pinggu model/1111result/1111CAM'
CAM_output_binary = 'Pinggu model/1111result/1111binary'
model_path = 'Pinggu model/1111result/1111CAM_model.pth'
################################################################


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

        img = torch.from_numpy(img).unsqueeze(0)  
       
        num_classes = len(self.classes)
        one_hot_target = torch.zeros(num_classes)
        one_hot_target[target] = 1

        return img,one_hot_target


def returnCAM(feature_conv, weight_softmax, class_idx):
    size_upsample = (128, 128)
    bz, nc, h, w = feature_conv.shape
    output_cam = []
    for idx in class_idx:
        if idx == 0:
         cam = weight_softmax[idx].dot(feature_conv.reshape((nc, h*w)))
         cam = cam.reshape(h, w)
         cam = cam - np.min(cam)
         cam_img = cam / np.max(cam)
         alpha = 0.9  
         cam_img = np.clip(cam_img, 0, alpha)
         cam_img = np.uint8(255 * cam_img)
         cam_img = 255 - cam_img
         output_cam.append(cv2.resize(cam_img, size_upsample))
    return output_cam


normalize = transforms.Normalize(
   mean=[0.485, 0.456, 0.406],
   std=[0.229, 0.224, 0.225]
)


preprocess = transforms.Compose([
   transforms.Resize((224,224)),
   transforms.ToTensor(),
   normalize
])



model = mymodel(in_channel=1)



params = list(model.parameters()) 
weight_softmax = np.squeeze(params[-1].data.cpu().numpy())
weight_softmax = weight_softmax.reshape((1, 2))

for folder in [CAM_output_folder, CAM_output_binary]:
    if not os.path.exists(folder):
        os.makedirs(folder, exist_ok=True)



model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()


with open(LABELS_file) as f:
    classes = json.load(f)



for filename in os.listdir(image_file_path):
    features_blobs = []

    def hook_feature(module, input, output):  
        features_blobs.append(output.data.cpu().numpy())  

    model.final_cam.register_forward_hook(hook_feature)  

    if filename.endswith('.npy'): 
        image_file = os.path.join(image_file_path, filename)
        img_np = np.load(image_file)
        img_np = img_np.astype(np.float32)
        img_np = img_np[:176, :, :]  
        img_tensor = torch.from_numpy(img_np).unsqueeze(0)
        img_variable = Variable(img_tensor.unsqueeze(0))
        img_variable = Variable(img_tensor.unsqueeze(0))  
        logit = model(img_variable)
        logit = logit.view(1, 2)
        h_x = F.softmax(logit, dim=1).data.squeeze()
        probs, idx = h_x.sort(0, True)
        probs = probs.numpy()
        idx = idx.numpy()

    
        for i in range(0, 2):
            print('{:.3f} -> {}'.format(probs[i], classes[idx[i]]))

        if classes[idx[0]] == "crack":
            CAMs = returnCAM(features_blobs[0], weight_softmax, [idx[0]])

            print('output CAM.jpg for the top1 prediction: %s' % classes[idx[0]])

            red_band = img_np[71, :, :]  
            green_band = img_np[42, :, :]  
            blue_band = img_np[20, :, :]  
            rgb_image = np.dstack((red_band, green_band, blue_band))

        
            heatmap = cv2.applyColorMap(cv2.resize(CAMs[0], rgb_image.shape[:2][::-1]), cv2.COLORMAP_JET)

            result = heatmap * 0.3 + rgb_image * 0.5  
            output_path = os.path.join(CAM_output_folder, f'{filename.split(".")[0]}.png')
            cv2.imwrite(output_path, result)

            cam_image = np.uint8(CAMs[0])

        
        
            _, binary_heatmap = cv2.threshold(cam_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            binary_heatmap = 255 - binary_heatmap

            binary_output_path = os.path.join(CAM_output_binary, f'{filename.split(".")[0]}.png')
            cv2.imwrite(binary_output_path, binary_heatmap)