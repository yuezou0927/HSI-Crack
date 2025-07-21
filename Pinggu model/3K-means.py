import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import os


####################################################################
cam_folder = "Pinggu model/1111result/1111CAM"
image_folder = "crack_data_Pinggu/train/traindata/0"  
output_folder = "Pinggu model/1111result/1111kmeans"
######################################################################

if not os.path.exists(output_folder):
    os.makedirs(output_folder)


all_selected_bands = []
valid_filenames = []  
image_shapes = [] 


for filename in os.listdir(image_folder):
    if filename.endswith('.npy'):
   
        image_file = os.path.join(image_folder, filename)
        image_data = np.load(image_file)

        image_data = image_data.transpose(1, 2, 0)
        selected_bands = image_data[:, :, :176]
        print(f"选取的波段: {selected_bands.shape}")

        all_selected_bands.append(selected_bands)
        valid_filenames.append(filename)
        image_shapes.append(selected_bands.shape)



num_valid_files = len(valid_filenames)
print(f"参与聚类的有效文件数量: {num_valid_files}")


all_selected_bands = np.concatenate(all_selected_bands, axis=0)



height, width, bands = all_selected_bands.shape
data_reshaped = all_selected_bands.reshape(-1, bands)


scaler = StandardScaler()
data_scaled = scaler.fit_transform(data_reshaped)


k = 11  
kmeans = KMeans(n_clusters=k, random_state=64, n_init=10)
labels = kmeans.fit_predict(data_scaled)


start_idx = 0
for i, (filename, shape) in enumerate(zip(valid_filenames, image_shapes)):
  
    img_height, img_width, _ = shape
    pixel_count = img_height * img_width

    end_idx = start_idx + pixel_count
    img_labels = labels[start_idx:end_idx].reshape(img_height, img_width)
    start_idx = end_idx
    

    image_name = filename.split('.')[0]
    cam_file = os.path.join(cam_folder, f"{image_name}.png")
    if os.path.exists(cam_file):

        output_kmeans_path = os.path.join(output_folder, f"{image_name}.png")
        plt.imsave(output_kmeans_path, img_labels, cmap='tab10')
        print(f"已保存聚类结果: {output_kmeans_path}")


print("所有聚类结果处理完成！")