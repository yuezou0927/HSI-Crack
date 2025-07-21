import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from collections import Counter

############################################################
cam_folder = "Leeds model/1111result/1111CAM"
kmeans_folder = "Leeds model/1111result/1111kmeans"
cam_binary_folder = "Leeds model/1111result/1111binary"
output_folder = "Leeds model/1111result/1111pse-label"
###########################################################


if not os.path.exists(output_folder):
    os.makedirs(output_folder)


max_prob_classes = []


for cam_filename in os.listdir(cam_folder):
    if cam_filename.endswith('.png'):
        base_name = os.path.splitext(cam_filename)[0]
        kmeans_filename = f"{base_name}.png"
        kmeans_file_path = os.path.join(kmeans_folder, kmeans_filename)
       
        cam_binary_filename = f"{base_name}.png"
        cam_binary_file_path = os.path.join(cam_binary_folder, cam_binary_filename)

        if os.path.exists(kmeans_file_path) and os.path.exists(cam_binary_file_path):
            
            cam_image = cv2.imread(os.path.join(cam_folder, cam_filename), cv2.IMREAD_GRAYSCALE)

            #cam_image = 255 - cam_image
            cam_prob = cam_image / 255.0  

            kmeans_image = cv2.imread(kmeans_file_path, cv2.IMREAD_GRAYSCALE)

            num_classes = np.max(kmeans_image)

            class_prob_sum = np.zeros(num_classes + 1)
            class_pixel_count = np.zeros(num_classes + 1)

            height, width = kmeans_image.shape
            for i in range(height):
                for j in range(width):
                    class_id = kmeans_image[i, j]
                    class_prob_sum[class_id] += cam_prob[i, j]
                    class_pixel_count[class_id] += 1

            class_avg_prob = class_prob_sum / (class_pixel_count + 1e-8)  

            max_prob_class = np.argmax(class_avg_prob)
            max_prob_classes.append(max_prob_class)

            print(f"文件名: {base_name}")
            for class_id in range(num_classes + 1):
                if class_avg_prob[class_id] > 0:
                    print(f"类别 {class_id} 的平均概率: {class_avg_prob[class_id]}")
            print(f"最大平均概率的类别: {max_prob_class}")


counter = Counter(max_prob_classes)

most_common_class = counter.most_common(1)[0][0]
print(f"所有图中出现次数最多的最大平均概率类别: {most_common_class}")


for cam_filename in os.listdir(cam_folder):
    if cam_filename.endswith('.png'):
        base_name = os.path.splitext(cam_filename)[0]
        kmeans_filename = f"{base_name}.png"
        kmeans_file_path = os.path.join(kmeans_folder, kmeans_filename)
        cam_binary_filename = f"{base_name}.png"
        cam_binary_file_path = os.path.join(cam_binary_folder, cam_binary_filename)

        if os.path.exists(kmeans_file_path) and os.path.exists(cam_binary_file_path):
           
            kmeans_image = cv2.imread(kmeans_file_path, cv2.IMREAD_GRAYSCALE)
      
            cam_binary_image = cv2.imread(cam_binary_file_path, cv2.IMREAD_GRAYSCALE)

            processed_binary_image = np.ones_like(cam_binary_image) * 0

            height, width = kmeans_image.shape
            for i in range(height):
                for j in range(width):
                    if cam_binary_image[i, j] == 255:  
                        if kmeans_image[i, j] == most_common_class:
                            processed_binary_image[i, j] = 255

            output_path = os.path.join(output_folder, f"{base_name}.png")
            cv2.imwrite(output_path, processed_binary_image)