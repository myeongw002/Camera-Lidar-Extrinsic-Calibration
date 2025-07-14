import numpy as np
from sensor_msgs.msg import PointCloud2
import open3d as o3d
import cv2
import os
import glob
from copy import deepcopy
import random
from utils import *
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation 
import matplotlib.pyplot as plt

base_dir = "./result1"

RESULT_DIRS = [
    base_dir + "/unoptimized_plane",
    base_dir + "/unoptimized_pcd",
    base_dir + "/iou",
    base_dir + "/iou_optimized",
    base_dir + "/iou_optimized_plane",
]

for d in RESULT_DIRS:
    os.makedirs(d, exist_ok=True)

# 체커보드 크기 설정 (내부 코너 개수)
CHECKERBOARD = (5, 7)  # (가로, 세로)
scale = 0.095  # 체커보드 크기 (m)

# 체커보드 3D 좌표 생성
objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
objp = objp * scale  # 크기 적용
#print("Checkerboard 3D Points:\n", objp)

padding_x = 0.015  # 가로 방향 패딩 (1.05cm)
padding_y = 0.065  # 세로 방향 패딩 (2.1cm)

nx, ny = CHECKERBOARD          # (5, 7)

# 네 모서리의 내부 코너 인덱스
idx_TL = 0                     # (0, 0)
idx_TR = nx - 1                # (nx‑1, 0)
idx_BR = nx*ny - 1             # (nx‑1, ny‑1)
idx_BL = nx*(ny-1)             # (0, ny‑1)

board_corners = np.array([
    objp[idx_TL] + [-scale - padding_x,  -scale - padding_y, 0],   # 좌‑상단
    objp[idx_TR] + [ scale + padding_x,  -scale - padding_y, 0],   # 우‑상단
    objp[idx_BR] + [ scale + padding_x,   scale + padding_y, 0],   # 우‑하단
    objp[idx_BL] + [-scale - padding_x,   scale + padding_y, 0],   # 좌‑하단
], dtype=np.float32)

# 데이터 경로 설정
image_path = "extrinsic_data/valid_images"
pointcloud_path = "extrinsic_data/valid_pointclouds"
intrinsic_path = "extrinsic_data/camera/intrinsic1.csv"

# 카메라 내부 파라미터 로드
intrinsic_param = np.loadtxt(intrinsic_path, delimiter=',', usecols=range(9))
intrinsic = np.array([[intrinsic_param[0], intrinsic_param[1], intrinsic_param[2]],
                      [0.0, intrinsic_param[3], intrinsic_param[4]],
                      [0.0, 0.0, 1.0]])
distortion = np.array(intrinsic_param[5:9])

print("Camera Matrix:\n", intrinsic)
print("Distortion Coefficients:\n", distortion)

# 이미지 & 포인트 클라우드 정렬
image_files = sorted(glob.glob(os.path.join(image_path, "*.jpg")) + glob.glob(os.path.join(image_path, "*.png")))
pointcloud_files = sorted(glob.glob(os.path.join(pointcloud_path, "*.pcd")))
image_files.sort()
pointcloud_files.sort()

# 유효한 이미지 및 포인트 클라우드 개수
print("Number of valid images:", len(image_files))
print("Number of valid point clouds:", len(pointcloud_files))
# Open3D에서 시각화할 객체 리스트
#geometry_list = []

# 카메라 좌표축 추가 (원점에서 시작)
# Open3D 카메라 좌표축 생성
camera_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2)
#geometry_list.append(camera_frame)

show_image = False
show_o3d = False

initial_guess_list = []
pcd_list = [] 
img_list = []

corner_list = []    
image_T_list = []
print("Finding checkerboard corners in images...")
for img_idx, file in enumerate(image_files):
    image = cv2.imread(file)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 체커보드 코너 찾기
    ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, None)

    if not ret:
        print(f"Corner not found idx: {img_idx}")

    if ret:
        geometry_list = []
        geometry_list.append(camera_frame)
        # 코너 정밀화
        # print("체커보드 코너 좌표:\n", corners.shape)
        corners = corners.reshape(-1, 2)  # 2D 좌표로 변환
        # print("체커보드 코너 좌표:\n", corners)
        # print(corners[0][1], corners[-1][1])

        if corners[0][1] > corners[-1][1]:
            corners = np.flip(corners, axis=0)
            #print("체커보드 코너 좌표 뒤집힘")

        # 다시 OpenCV가 사용할 수 있도록 형태 변경
        corners = corners.reshape(-1, 1, 2).astype(np.float32)
        # print("체커보드 코너 좌표:\n", corners.shape)
        # print("체커보드 코너 좌표 (OpenCV 형식):\n", corners)
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), 
                                    criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))

        # SolvePnP로 회전 및 이동 벡터 계산
        _, rvec, tvec = cv2.solvePnP(objp, corners2, intrinsic, distortion)

        if show_image:
            # draw axis
            # 체커보드 코너 그리기
            cv2.drawChessboardCorners(image, CHECKERBOARD, corners, ret)
            cv2.drawFrameAxes(image, intrinsic, distortion, rvec, tvec, length=0.1, thickness=2)
            cv2.imshow("Checkerboard Projection", image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        # 회전 행렬 계산
        R, _ = cv2.Rodrigues(rvec)
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = tvec.flatten()
        
        checkerboard_frame = deepcopy(camera_frame)  # 카메라 좌표계 복사
        checkerboard_frame.transform(T)  # 변환 적용
        geometry_list.append(checkerboard_frame)

        # 체커보드 코너를 Open3D 포인트로 변환
        objp_homogeneous = np.hstack((objp, np.ones((objp.shape[0], 1))))  # 3D → 4D (Homogeneous)
        objp_transformed = (T @ objp_homogeneous.T).T[:, :3]  # 변환 적용 후 3D 좌표만 사용

        # Open3D 포인트 클라우드 생성
        objp_pcd = o3d.geometry.PointCloud()
        objp_pcd.points = o3d.utility.Vector3dVector(objp_transformed)  # 변환된 좌표 사용
        objp_pcd.paint_uniform_color([1, 0, 0])  # 빨간색 (체커보드 코너)
        geometry_list.append(objp_pcd)
        checkerboard_normal = R[:,2]  # 체커보드의 법선 벡터
        print("체커보드 법선 벡터:", checkerboard_normal)
        # 바운딩 박스 설정 (체커보드 크기 기반)
        bbox_min = objp_transformed.min(axis=0) - np.array([1.0, 1.0, 1.0])  # 여유 공간 추가
        bbox_max = objp_transformed.max(axis=0) + np.array([1.0, 1.0, 1.0])

        # Open3D AABB (축 정렬된 바운딩 박스) 생성
        aabb = o3d.geometry.AxisAlignedBoundingBox(bbox_min, bbox_max)
        aabb.color = (1, 0, 0)  # 빨간색 바운딩 박스
        geometry_list.append(aabb)

        # 포인트 클라우드 로드
        pcd_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2)
        pcd = o3d.io.read_point_cloud(pointcloud_files[img_idx]) 
        
        rot_vec_x = np.array([[np.pi/2], [0], [0]]) # x:90 y:0 z:0 deg
        rot_vec_z = np.array([[0], [0], [np.pi/2]]) # x:0 y:0 z:90 deg
        rotation_matrix_x,_ = cv2.Rodrigues(rot_vec_x)
        rotation_matrix_z,_ = cv2.Rodrigues(rot_vec_z)
        rotation_matrix = rotation_matrix_x @ rotation_matrix_z
        translation_vector = np.array([0, 0, 0])
        # 변환 행렬 생성
        init_transform = np.eye(4)
        init_transform[:3, :3] = rotation_matrix
        init_transform[:3, 3] = translation_vector
        
        # 포인트 클라우드 변환
        pcd_frame.transform(init_transform)
        geometry_list.append(pcd_frame)
        pcd.transform(init_transform)
        geometry_list.append(pcd)

        if show_o3d:
            o3d.visualization.draw_geometries(geometry_list, window_name=f"PointCloud {img_idx}", width=800, height=600)

        filtered_pcd = pcd.crop(aabb)

        if show_o3d:
            o3d.visualization.draw_geometries([filtered_pcd], window_name=f"Filtered PointCloud {img_idx}", width=800, height=600)
        
        # Dbscan 클러스터링 적용
        eps = 0.05  # 클러스터 간 거리 임계값
        min_points = 5  # 최소 클러스터 크기
        labels = np.array(filtered_pcd.cluster_dbscan(eps=eps, min_points=min_points, print_progress=True))
        num_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        print(f"Detected {num_clusters} clusters.")
        # 각 클러스터에 색상 할당
        max_label = labels.max()
        colors = np.array([random.choices(range(256), k=3) for _ in range(max_label + 1)]) / 255.0  # RGB 색상 생성
        colors = np.vstack([[0, 0, 0], colors])  # 노이즈 포인트는 검은색
        # 포인트 클라우드에 색상 적용
        filtered_pcd.colors = o3d.utility.Vector3dVector(colors[labels + 1])
        # 노이즈 포인트 제거
        # 노이즈 포인트는 검은색으로 설정
        #filtered_pcd.colors = o3d.utility.Vector3dVector(colors[labels + 1])
        # Open3D 시각화 실행
        if show_o3d:
            o3d.visualization.draw_geometries([filtered_pcd], window_name=f"DBSCAN Clustering {img_idx}", width=800, height=600)
    

        best_plane = None
        best_dot_product = -1
        best_plane_pcd = None
        dist_th = 0.03  # 평면 검출 거리 임계값
        ransac_n = 3  # RANSAC에서 사용할 점 개수
        num_iterations = 1000  # RANSAC 반복 횟수

        for cluster_id in np.unique(labels):
            if cluster_id == -1:  # 노이즈 제거
                continue
            
            cluster_indices = np.where(labels == cluster_id)[0]
    
            if len(cluster_indices) < 3:  # 최소 RANSAC 점 개수보다 작은 클러스터는 무시
                # print(f"Skipping cluster {cluster_id} (size: {len(cluster_indices)} points)")
                continue

            cluster_pcd = filtered_pcd.select_by_index(np.where(labels == cluster_id)[0])
            # o3d.visualization.draw_geometries([cluster_pcd], window_name="Clustered PointCloud")
            # RANSAC을 이용한 평면 검출
            plane_model, inliers = cluster_pcd.segment_plane(dist_th, ransac_n, num_iterations)
            if len(inliers) < 50:
                continue

            plane_pcd = cluster_pcd.select_by_index(inliers)
            plane_pcd.paint_uniform_color([random.random(), random.random(), random.random()])  # 랜덤 색상

            plane_normal = np.array(plane_model[:3])
            dot_product = np.abs(np.dot(plane_normal, checkerboard_normal))

            # print(f"Cluster {cluster_id} - Plane normal: {plane_normal}, Dot product: {dot_product}")

            # 체커보드와 가장 평행한 평면 선택
            if dot_product > best_dot_product:
                best_dot_product = dot_product
                best_plane = plane_model
                best_plane_pcd = plane_pcd

        if best_dot_product < 0.9:
            print(f"No suitable plane found for image {img_idx} (dot product: {best_dot_product})")
            continue

        num_points = len(best_plane_pcd.points)
        print(f"Best plane found for image {img_idx} with {num_points} points")

        if num_points < 200:
            print(f"Best plane has too few points ({num_points}), skipping image {img_idx}")
            continue

        # 선택된 평면을 강조하여 시각화
        if best_plane_pcd:
            best_plane_pcd.paint_uniform_color([0, 1, 0])  # 초록색 (체커보드와 평행한 평면)
            geometry_list.append(best_plane_pcd)
            # print(f"Best plane selected with normal {best_plane[:3]} (dot product: {best_dot_product})")
            
        # Open3D 시각화 실행
        if show_o3d :
            o3d.visualization.draw_geometries(geometry_list, window_name=f"Plane Detection {img_idx}")
            o3d.visualization.draw_geometries([best_plane_pcd, objp_pcd], window_name=f"Best Plane {img_idx}", width=800, height=600)

        # 체커보드와 평행한 평면 필터링
        radius = 0.6  # 필터링 반경
        filtered_plane = filter_points_inside_chessboard(best_plane_pcd, radius)
        # 필터링된 포인트 클라우드 시각화
        if show_o3d:
            o3d.visualization.draw_geometries([filtered_plane], window_name=f"Filtered Plane {img_idx}", width=800, height=600)
        # ICP 수행
        camera_center = np.mean(np.asarray(objp_pcd.points), axis=0)  # 체커보드의 중심
        # LiDAR에서 검출된 체커보드 정보
        lidar_center = np.mean(np.asarray(filtered_plane.points), axis=0)  # LiDAR 체커보드의 중심
        translate_vector = camera_center - lidar_center
        icp_init_transform = np.eye(4)
        icp_init_transform[:3, 3] = translate_vector
        # 초기 변환 행렬 계산
        # print("Camera center:", camera_center)
        # print("LiDAR center:", lidar_center)
        # print("Initial transformation matrix:\n", init_transform)
        # ICP 수행
        unoptimized_image = image.copy()
        lidar_to_cam = perform_icp(camera_frame, pcd_frame, filtered_plane, objp_pcd, threshold=0.05, init_transform=icp_init_transform, visualize=show_o3d)
        # print("best plane pcd", best_plane_pcd)
        unoptimized_projection_img, projected_points = pcd_projection(unoptimized_image, filtered_plane, intrinsic, distortion, lidar_to_cam)
        pcd_list.append(filtered_plane)
        img_list.append(file)
        result_path = RESULT_DIRS[0] + f"/unptimized_plane_{img_idx}.jpg"
        print(f"Projected image saved at: {result_path}")
        cv2.imwrite(result_path, unoptimized_projection_img)

        unoptimized_image2 = image.copy()
        unoptimized_projection_img2, _ = pcd_projection(unoptimized_image2, pcd, intrinsic, distortion, lidar_to_cam)
        result_path = RESULT_DIRS[1] + f"/unptimized_pcd_{img_idx}.jpg"
        print(f"Projected image saved at: {result_path}")
        cv2.imwrite(result_path, unoptimized_projection_img2)

        countour_img, contour_points = draw_contour(unoptimized_projection_img, board_corners, rvec, tvec, intrinsic, distortion)
        corner_list.append(board_corners)
        rvec_init, _ = cv2.Rodrigues(lidar_to_cam[:3, :3])
        tvec_init = lidar_to_cam[:3, 3].reshape(-1, 1)
        # 최적화 초기값 설정
        initial_guess = np.hstack((rvec_init.flatten(), tvec_init.flatten()))
        initial_guess_list.append(initial_guess)
        iou_img = draw_iou(countour_img, projected_points, contour_points)
        cv2.imwrite(RESULT_DIRS[2] + f"/iou_image_{img_idx}.jpg", iou_img)
        print(f"IOU image saved at: ./result/iou_image/iou_image_{img_idx}.jpg")
        image_T_list.append(T)

        if show_image:
            cv2.imshow("IOU", iou_img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()


print("number of initial guesses:", len(initial_guess_list))
print("number of point clouds:", len(pcd_list))
print("number of corners:", len(corner_list))
print("number of image_T:", len(image_T_list))

rvecs = np.array([guess[:3] for guess in initial_guess_list])

# 2. 회전 벡터를 SciPy Rotation 객체로 변환
rotations = Rotation.from_rotvec(rvecs)

# 3. Quaternion 평균을 적용하여 평균 회전 계산
# trasnform matrix로 평균하면 직교성 깨짐 -> 회전 벡터 평균을 사용
mean_rotation = rotations.mean()
mean_rvec = mean_rotation.as_rotvec()  # 평균 회전 벡터

# 4. 각 이미지의 이동 벡터(tvec)를 추출하고 단순 평균 계산
tvecs = np.array([guess[3:] for guess in initial_guess_list])
mean_tvec = np.mean(tvecs, axis=0)

# 5. 평균 회전과 평균 이동을 결합한 초기 추정값 생성 (6차원 벡터)
averaged_initial_guess = np.hstack((mean_rvec, mean_tvec))

unit_transform = np.eye(4)
init_rvec,_ = cv2.Rodrigues(unit_transform[:3, :3])
init_tvec = unit_transform[:3, 3].reshape(-1, 1)
# 초기 변환 행렬을 6차원 벡터로 변환
initial_guess = np.hstack((init_rvec.flatten(), init_tvec.flatten()))

#select init param 
init_param = averaged_initial_guess

print("Strarting optimization with IOU errors...")
initial_cost = joint_iou_loss(init_param, pcd_list, corner_list, intrinsic, distortion, image_T_list)
initial_cost = np.sum(initial_cost**2)

iou_result = least_squares(
    joint_iou_loss, 
    init_param, 
    args=(pcd_list, corner_list, intrinsic, distortion, image_T_list), 
    method='lm',
    loss='linear',
    )

print("최적화 결과:", iou_result.x)
print("최적화 성공 여부:", iou_result.success)
print("Initial cost:", initial_cost)
print("Cost:", iou_result.cost)
print("Average of residuals:", np.mean(iou_result.fun))
print(iou_result.message)
# 최적화된 회전 및 이동 벡터
optimized_rvec = iou_result.x[:3]
optimized_tvec = iou_result.x[3:].reshape(-1, 1)
# 최적화된 회전 행렬
optimized_R, _ = cv2.Rodrigues(optimized_rvec)
optimized_T = np.eye(4)
optimized_T[:3, :3] = optimized_R
optimized_T[:3, 3] = optimized_tvec.flatten()
_optimized_T = optimized_T # 최적화된 변환 행렬
optimized_T = optimized_T @ init_transform  # 초기 변환 행렬과 결합
print("최적화된 변환 행렬:\n", optimized_T)
#save optimized transform matrix
print("Saving optimized transform matrix...")
np.savetxt("iou_optimized_transform.txt", optimized_T, delimiter=",")

# 최적화된 변환 행렬을 사용하여 포인트 클라우드 변환
for i in range(len(img_list)):
    image = cv2.imread(img_list[i])
    pcd = o3d.io.read_point_cloud(pointcloud_files[i])
    # 이미지에 최적화된 변환 적용
    optimized_image, _ = pcd_projection(image, pcd, intrinsic, distortion, optimized_T)
    result_path = RESULT_DIRS[3] + f"/iou_optimized_image_{i}.jpg"
    print(f"IOU Optimized projected image saved at: {result_path}")
    cv2.imwrite(result_path, optimized_image)

# 최적화된 평면 시각화
for i in range(len(img_list)):
    image = cv2.imread(img_list[i])
    pcd = pcd_list[i]
    #print("pcd:", pcd.has_points())
    image_T = image_T_list[i]
    image_rvec = image_T[:3, :3]
    image_rvec = cv2.Rodrigues(image_rvec)[0]
    image_tvec = image_T[:3, 3].reshape(-1, 1)
    # 이미지에 최적화된 변환 적용
    # print(board_corners)
    plane_projection_img, projected_points = pcd_projection(image, pcd, intrinsic, distortion, _optimized_T)
    # print("projected points:", projected_points)
    countour_img, contour_points = draw_contour(plane_projection_img, board_corners, image_rvec, image_tvec, intrinsic, distortion)
    optimized_image = draw_iou(countour_img, projected_points, contour_points)
    if optimized_image is None:
        continue
    result_path = RESULT_DIRS[4] + f"/iou_optimized_plane_image_{i}.jpg"
    print(f"IOU Optimized projected image saved at: {result_path}")
    cv2.imwrite(result_path, optimized_image)



plot_cost_history(iou_cost_history, "IOU Cost History")

print("최적화 결과:", iou_result.x)
print("최적화 성공 여부:", iou_result.success)
print("Initial cost:", initial_cost)
print("Cost:", iou_result.cost)
print("Average of residuals:", np.mean(iou_result.fun))


# print("Strarting optimization with reprojection errors...")

# initial_cost = reprojection_error(initial_guess, pcd_list, corner_list, intrinsic, distortion, image_T_list)
# initial_cost = np.sum(initial_cost**2)

# reprojection_result = least_squares(
#     reprojection_error, 
#     averaged_initial_guess, 
#     args=(pcd_list, corner_list, intrinsic, distortion, image_T_list), 
#     method='lm',
#     loss='linear',
#     ftol=1e-15, xtol=1e-15, gtol=1e-15, 
#     max_nfev=10000,
#     )

# print("최적화 결과:", reprojection_result.x)
# print("최적화 성공 여부:", reprojection_result.success)
# print("Initial cost:", initial_cost)
# print("Cost:", reprojection_result.cost)

# plot_cost_history(reprojection_cost_history, "Reprojection Cost History")

# # 최적화된 회전 및 이동 벡터
# optimized_rvec = reprojection_result.x[:3]
# optimized_tvec = reprojection_result.x[3:].reshape(-1, 1)
# # 최적화된 회전 행렬
# optimized_R, _ = cv2.Rodrigues(optimized_rvec)
# optimized_T = np.eye(4)
# optimized_T[:3, :3] = optimized_R
# optimized_T[:3, 3] = optimized_tvec.flatten()
# optimized_T = optimized_T @ init_transform  # 초기 변환 행렬과 결합
# print("최적화된 변환 행렬:\n", optimized_T)
# #save optimized transform matrix
# print("Saving optimized transform matrix...")
# np.savetxt("reprojection_optimized_transform.txt", optimized_T, delimiter=",")

# # 최적화된 변환 행렬을 사용하여 포인트 클라우드 변환
# for i in range(len(image_files)):
#     image = cv2.imread(image_files[i])
#     pcd = o3d.io.read_point_cloud(pointcloud_files[i])
#     # 이미지에 최적화된 변환 적용
#     optimized_image, _ = pcd_projection(image, pcd, intrinsic, distortion, optimized_T)
#     result_path = f"./result/reprojection_optimized/reprojection_optimized_image_{i}.jpg"
#     print(f"Reprojection Optimized projected image saved at: {result_path}")
#     cv2.imwrite(result_path, optimized_image)
