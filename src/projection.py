import cv2
import numpy as np
import glob
import os
import open3d as o3d
from utils import pcd_projection

image_path = "extrinsic_data/camera"
pointcloud_path = "extrinsic_data/ouster"
result_path = "result/projection_images"
intrinsic_path = "24252427/intrinsic.csv"
extrinsic_path = "iou_optimized_transform.txt"

image_files = sorted(glob.glob(os.path.join(image_path, "*.jpg")) + glob.glob(os.path.join(image_path, "*.png")))
pointcloud_files = sorted(glob.glob(os.path.join(pointcloud_path, "*.pcd")))
image_files.sort()
pointcloud_files.sort()

intrinsic_param = np.loadtxt(intrinsic_path, delimiter=',', usecols=range(9))
intrinsic = np.array([[intrinsic_param[0], intrinsic_param[1], intrinsic_param[2]],
                      [0.0, intrinsic_param[3], intrinsic_param[4]],
                      [0.0, 0.0, 1.0]])
distortion = np.array(intrinsic_param[5:])

print("Camera Matrix:\n", intrinsic)
print("Distortion Coefficients:\n", distortion)

extrinsic = np.loadtxt(extrinsic_path, delimiter=',')
# extrinsic 변환 행렬을 4x4로 변환  
print(extrinsic)

# 최적화된 변환 행렬을 사용하여 포인트 클라우드 변환
for i in range(len(image_files)):
    image = cv2.imread(image_files[i])
    pcd = o3d.io.read_point_cloud(pointcloud_files[i])
    # 이미지에 최적화된 변환 적용
    optimized_image, _ = pcd_projection(image, pcd, intrinsic, distortion, extrinsic)
    image_path = result_path + f"/image_{i}.jpg"
    print(f"Projected image saved at: {image_path}")
    cv2.imwrite(image_path, optimized_image)
