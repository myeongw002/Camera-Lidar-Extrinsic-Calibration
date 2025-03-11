import open3d as o3d
import numpy as np
import random

# PCD 파일 로드
pcd_path = "extrinsic_data/valid_pointclouds/0003.pcd"  # PCD 파일 경로
pcd = o3d.io.read_point_cloud(pcd_path)

# DBSCAN 클러스터링 적용
eps = 0.1  # 한 포인트가 다른 포인트와 같은 클러스터에 속하기 위한 거리 임계값 (m)
min_points = 10  # 클러스터가 되기 위한 최소 포인트 개수

labels = np.array(pcd.cluster_dbscan(eps=eps, min_points=min_points, print_progress=True))

# 클러스터 개수 및 노이즈 출력
num_clusters = len(set(labels)) - (1 if -1 in labels else 0)
print(f"Detected {num_clusters} clusters")

# 각 클러스터에 색상 할당
max_label = labels.max()
colors = np.array([random.choices(range(256), k=3) for _ in range(max_label + 1)]) / 255.0  # RGB 색상 생성
colors = np.vstack([[0, 0, 0], colors])  # 노이즈 포인트는 검은색

# 포인트 클라우드에 색상 적용
pcd.colors = o3d.utility.Vector3dVector(colors[labels + 1])

# Open3D 시각화 실행
o3d.visualization.draw_geometries([pcd], window_name="DBSCAN Clustering")
