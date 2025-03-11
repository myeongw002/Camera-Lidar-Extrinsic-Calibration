import rospy
import numpy as np
from sensor_msgs.msg import PointCloud2
import open3d as o3d
import cv2
import os
import glob

# 체커보드 크기 설정 (내부 코너 개수)
CHECKERBOARD = (4, 5)  # (가로, 세로)

# 원본 이미지 및 포인트 클라우드 경로
image_path = "extrinsic_data/camera"
pointcloud_path = "extrinsic_data/ouster"

# 유효한 데이터를 저장할 폴더
valid_image_path = "extrinsic_data/valid_images"
valid_pointcloud_path = "extrinsic_data/valid_pointclouds"

os.makedirs(valid_image_path, exist_ok=True)  # 폴더 생성
os.makedirs(valid_pointcloud_path, exist_ok=True)  # 폴더 생성

# 이미지 & PointCloud 파일 읽기 및 정렬
image_files = sorted(glob.glob(os.path.join(image_path, "*.jpg")) + glob.glob(os.path.join(image_path, "*.png")))
pointcloud_files = sorted(glob.glob(os.path.join(pointcloud_path, "*.pcd")))  # 포인트 클라우드 파일 정렬

valid_image_files = 0
invalid_image_files = 0
valid_idx = []  # 유효한 이미지 여부를 저장하는 리스트

show_image = False  # 이미지 표시 여부

print("Finding checkerboard corners in images...")
# 이미지 처리 및 체커보드 감지
for img_idx, file in enumerate(image_files):
    image = cv2.imread(file)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 체커보드 코너 찾기
    ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, None)

    if ret:
        # 코너 정밀화
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), 
                                     criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
        cv2.drawChessboardCorners(gray, CHECKERBOARD, corners2, ret)
        
        valid_image_files += 1
        valid_idx.append(1)  # 유효한 이미지 표시

        #print(f"체커보드 감지됨: {file}")

        # 유효한 이미지 저장
        valid_image_file = os.path.join(valid_image_path, os.path.basename(file))
        cv2.imwrite(valid_image_file, image)
    else:
        invalid_image_files += 1
        valid_idx.append(0)  # 유효하지 않은 이미지 표시
        #print(f"체커보드 없음: {file}")

    # 결과 이미지 표시
    if show_image:
        cv2.imshow("Checkerboard Detection", image)
        cv2.waitKey(300)  # 0: 키 입력 대기
        cv2.destroyAllWindows()

print(f"유효한 이미지 파일 수: {valid_image_files}")
print(f"유효하지 않은 이미지 파일 수: {invalid_image_files}")

# ==============================
# 유효한 PointCloud 저장
# ==============================
valid_pc_count = 0

for idx, valid in enumerate(valid_idx):
    if valid == 1 and idx < len(pointcloud_files):  # 유효한 이미지이고, 인덱스가 유효하면
        pc_file = pointcloud_files[idx]  # 동일한 인덱스의 PointCloud 파일 선택
        pc_filename = os.path.basename(pc_file)
        valid_pc_file = os.path.join(valid_pointcloud_path, pc_filename)

        # Open3D를 사용하여 PointCloud 읽고 저장
        pointcloud = o3d.io.read_point_cloud(pc_file)
        o3d.io.write_point_cloud(valid_pc_file, pointcloud)

        valid_pc_count += 1
        #print(f"PointCloud 저장됨: {valid_pc_file}")

print(f"유효한 PointCloud 파일 수: {valid_pc_count}")


