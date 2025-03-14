import numpy as np
import open3d as o3d
import cv2
from copy import deepcopy
from scipy.spatial import ConvexHull
from sklearn.decomposition import PCA
from scipy.spatial import distance
import itertools

iou_cost_history = []
reprojection_cost_history = []



def perform_icp(camera_frame, pcd_frame, source, target, threshold, init_transform, visualize):
    # ICP 수행
    print("Performing ICP...")

    target.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.05, max_nn=30))
    reg_icp = o3d.pipelines.registration.registration_icp(
        source, target, threshold, init_transform,
        o3d.pipelines.registration.TransformationEstimationPointToPoint())
    print("ICP fitness:", reg_icp.fitness)
    print("ICP inlier RMSE:", reg_icp.inlier_rmse)
    print("ICP transformation:\n", reg_icp.transformation)
    source_cp = deepcopy(source)  # 원본 포인트 클라우드 복사
    source_cp.transform(reg_icp.transformation)  # 변환 적용
    pcd_frame.transform(reg_icp.transformation)  # 포인트 클라우드 변환
    if visualize:
        o3d.visualization.draw_geometries([camera_frame, pcd_frame, source_cp, target], window_name="ICP Result")
    # 변환 행렬 반환
    # print("Transformation matrix:\n", reg_icp.transformation)
    return reg_icp.transformation

def pcd_projection(img, pcd, intrinsic, distortion, transform, point_size=3, color=None):
    """
    포인트 클라우드를 2D 이미지 평면에 투영하고, 거리별 색상을 적용하는 함수.
    
    Args:
        img: 투영 대상 이미지 (numpy 배열, BGR 형식)
        pcd: Open3D 포인트 클라우드 객체
        intrinsic: 카메라 내부 파라미터 (3x3 행렬)
        distortion: 카메라 왜곡 계수 (1D 배열)
        transform: 포인트 클라우드 변환 행렬 (4x4)
        point_size: 투영된 점의 크기 (기본값: 3)
        color: 사용자 지정 색상 (BGR, shape=(3,) or (N,3)), None이면 거리 기반 색상 사용
    Returns:
        img: 투영 결과 이미지
        valid_points: 투영된 2D 좌표 리스트
    """
    # print("PCD : ", pcd)
    if not pcd.has_points():
        print("PCD:" , pcd)
        print("No points in the point cloud.")
        return img, []

    # LiDAR 포인트 변환
    pcd_points = np.asarray(pcd.points)
    pcd_homogeneous = np.hstack((pcd_points, np.ones((pcd_points.shape[0], 1))))  # 3D → 4D
    transformed_pcd = (transform @ pcd_homogeneous.T).T[:, :3]  # 변환 후 3D 좌표

    # 카메라 앞쪽에 있는 점만 필터링
    valid_mask = transformed_pcd[:, 2] > 0
    transformed_pcd = transformed_pcd[valid_mask]
    if transformed_pcd.shape[0] == 0:
        print("No points in front ofthe camera after transformation.")
        print("Transformed PCD:", transformed_pcd)
        return img, []

    # 포인트 별 거리 계산
    distances = np.linalg.norm(transformed_pcd, axis=1)

    # 거리 정규화 및 컬러맵 적용
    norm_distances = cv2.normalize(distances, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    norm_distances = norm_distances.astype(np.uint8)
    jet_colors = cv2.applyColorMap(norm_distances, cv2.COLORMAP_JET)  # JET 컬러맵 적용

    # 사용자 지정 색상 적용 (기본값은 거리 기반 JET 컬러)
    if color is None:
        colors = jet_colors
    else:
        color = np.array(color, dtype=np.uint8)
        if color.ndim == 1:
            colors = np.tile(color, (len(distances), 1))
        else:
            colors = color

    # LiDAR 포인트를 2D 이미지에 투영
    rvec = np.zeros((3, 1), dtype=np.float32)  # 회전 없음
    tvec = np.zeros((3, 1), dtype=np.float32)  # 이동 없음
    projected_points, _ = cv2.projectPoints(transformed_pcd, rvec, tvec, intrinsic, distortion)
    projected_points = projected_points.reshape(-1, 2)  # (N, 1, 2) → (N, 2)
    # print(f"Projected points shape: {projected_points.shape}")
    # print(f"Projected points: {projected_points}")
    # 이미지 범위 내 포인트 필터링
    h, w = img.shape[:2]
    valid_mask = (projected_points[:, 0] >= 0) & (projected_points[:, 0] < w) & \
                 (projected_points[:, 1] >= 0) & (projected_points[:, 1] < h)
    # print(f"Valid points: {np.sum(valid_mask)} / {len(valid_mask)}")
    valid_points = projected_points[valid_mask].astype(int)
    valid_colors = colors[valid_mask]

    # 이미지에 포인트 투영 (점 크기 조절)
    for (x, y), c in zip(valid_points, valid_colors):
        cv2.circle(img, (x, y), point_size, tuple(map(int, c.squeeze())), -1)  # -1: 원을 채움

    # print(f"Projected {len(valid_points)} points onto the image.")

    return img, valid_points


def draw_contour(img, corners, rvec, tvec, intrinsic, distortion):
    # 체커보드 코너를 2D 이미지 평면으로 투영
    projected_corners, _ = cv2.projectPoints(corners, rvec, tvec, intrinsic, distortion)
    projected_corners = projected_corners.reshape(-1, 2)  # (N, 1, 2) -> (N, 2)

    # 이미지에 체커보드 코너 그리기
    for pt_idx, pt in enumerate(projected_corners):
        # print(f"Point {pt_idx}: {pt}, type: {type(pt)}")
        cv2.circle(img, (int(pt[0]), int(pt[1])), 5, (0, 0, 255), -1)  # 초록색 점

    return img, projected_corners

def filter_points_inside_chessboard(lidar_pcd, radius=0.8):
    """
    체스보드 영역 내에 포함되지 않는 LiDAR 포인트를 필터링하는 함수.
    
    Args:
        lidar_pcd: LiDAR 포인트 클라우드 (Open3D PointCloud 객체)
    
    Returns:
        filtered_pcd: 체스보드 영역 내부의 포인트 클라우드
    """
    pcd_centroid = np.mean(np.asarray(lidar_pcd.points), axis=0)  # 포인트 클라우드의 중심 계산
    # filter 기준: 체스보드 영역의 중심을 기준으로 반지름 1.5m 이내의 포인트
    
    filtered_points = []
    for point in lidar_pcd.points:
        distance = np.linalg.norm(point - pcd_centroid)
        if distance < radius:
            filtered_points.append(point)
    # 필터링된 포인트로 새로운 포인트 클라우드 생성
    filtered_pcd = o3d.geometry.PointCloud()
    filtered_pcd.points = o3d.utility.Vector3dVector(filtered_points)
    filtered_pcd.paint_uniform_color([0, 1, 0])

    return filtered_pcd

def sort_points_by_angle(points, center):
    """점들을 중심에서 방위각으로 정렬"""
    angles = np.arctan2(points[:, 1] - center[1], points[:, 0] - center[0])
    sorted_indices = np.argsort(angles)
    return points[sorted_indices]

def reprojection_error(params, pcd_list, corner_list, intrinsic, distortion, image_T_list):
    residuals = []
    for pcd_pts, corner_pts, image_T in zip(pcd_list, corner_list, image_T_list):
        # 파라미터 설정
        rvec = params[:3].reshape(3, 1)
        tvec = params[3:6].reshape(3, 1)
        image_R = image_T[:3, :3]
        image_rvec = cv2.Rodrigues(image_R)[0]
        image_tvec = image_T[:3, 3].reshape(3, 1)

        # LiDAR 포인트 투영
        pcd_np = np.asarray(pcd_pts.points)
        projected_pcd, _ = cv2.projectPoints(pcd_np, rvec, tvec, intrinsic, distortion)
        projected_pcd = projected_pcd.reshape(-1, 2)

        # 체스보드 꼭지점 투영
        projected_corner, _ = cv2.projectPoints(corner_pts, image_rvec, image_tvec, intrinsic, distortion)
        projected_corner = projected_corner.reshape(-1, 2)

        # Convex Hull 계산
        hull = cv2.convexHull(projected_pcd.astype(np.float32)).reshape(-1, 2)

        # Convex Hull 중심과 점들 정렬
        hull_center = np.mean(hull, axis=0)
        sorted_hull = sort_points_by_angle(hull, hull_center)

        # 체스보드 꼭지점 정렬
        corner_center = np.mean(projected_corner, axis=0)
        sorted_corners = sort_points_by_angle(projected_corner, corner_center)

        # Convex Hull에서 체스보드 꼭지점에 대응하는 점 찾기
        matched_pts = []
        for corner in sorted_corners:
            dists = np.linalg.norm(sorted_hull - corner, axis=1)
            nearest_idx = np.argmin(dists)
            matched_pts.append(sorted_hull[nearest_idx])
        matched_pts = np.array(matched_pts)

        # 오차 계산
        distances = np.linalg.norm(sorted_corners - matched_pts, axis=1) / 4
        residuals.extend(distances)

    residuals = np.array(residuals)
    reprojection_cost_history.append(np.sum(residuals**2))

    return residuals

def compute_iou(lidar_poly, camera_poly):
    area_lidar = cv2.contourArea(lidar_poly)
    area_camera = cv2.contourArea(camera_poly)
    ret, intersection = cv2.intersectConvexConvex(lidar_poly, camera_poly)
    area_intersection = cv2.contourArea(intersection) if intersection is not None and len(intersection) > 0 else 0
    union = area_lidar + area_camera - area_intersection
    return area_intersection / (union + 1e-6)

def joint_iou_loss(params, pcd_list, corner_list, intrinsic, distortion, image_T_list):
    residuals = []
    for pcd_pts, corner_pts, imgae_T in zip(pcd_list, corner_list, image_T_list):
        # params: 6차원 extrinsic 파라미터 (회전, 이동)
        rvec = params[:3].reshape(3,1)
        tvec = params[3:6].reshape(3,1)

        image_R = imgae_T[:3,:3]
        image_rvec = cv2.Rodrigues(image_R)[0]
        image_tvec = imgae_T[:3,3].reshape(3,1)
        # 여기서는 각 이미지에 대해 이미 계산된 projected points(투영 결과)를 활용하거나,
        # 만약 재투영이 필요하다면 해당 LiDAR 포인트 클라우드를 params로 재투영하는 과정을 포함해야 합니다.
        # 예: projected_pts, _ = cv2.projectPoints(lidar_pts, rvec, tvec, intrinsic, distortion)
        pcd_np = np.asarray(pcd_pts.points)
        # corner_np = np.asarray(corner_pts.points)
        projected_pcd, _ = cv2.projectPoints(pcd_np, rvec, tvec, intrinsic, distortion)
        projected_corner, _ = cv2.projectPoints(corner_pts, image_rvec, image_tvec, intrinsic, distortion)
        
        # 예시로, convex hull을 구하고 IoU를 계산한다고 가정하면:
        hull = cv2.convexHull(projected_pcd.astype(np.float32)).reshape(-1,2)
        # print("Hull length:", len(hull))    
        iou = compute_iou(hull, projected_corner)
        iou_error = 1 - iou
        # print("IOU error:", iou_error)
        residuals.append(iou_error)

    residuals = np.array(residuals)
    iou_cost_history.append(np.sum(residuals**2))

    return residuals

def draw_iou(img, projected_points, contour_points):
    """
    IoU를 시각화하기 위한 함수
    Args:
        img: 이미지 (numpy 배열)
        lidar_poly: LiDAR 다각형 (numpy 배열)
        camera_poly: 카메라 다각형 (numpy 배열)
    Returns:
        img: IoU가 시각화된 이미지
    """
    hull = cv2.convexHull(projected_points.astype(np.float32)).reshape(-1,2).astype(np.int32)
    # LiDAR 다각형 그리기
    contour_points = contour_points.astype(np.int32)
    cv2.polylines(img, [hull], isClosed=True, color=(0, 255, 0), thickness=2)  # 초록색
    # 카메라 다각형 그리기
    cv2.polylines(img, [contour_points], isClosed=True, color=(0, 0, 255), thickness=2)  # 빨간색
    return img


def plot_cost_history(residuals, title="Residuals over Iterations"):
    """
    Residuals를 시각화하는 함수
    Args:
        residuals: Residual 값 리스트
        title: 그래프 제목
    """
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 6))
    plt.plot(residuals, marker='o')
    plt.title(title)
    plt.xlabel('Iteration')
    plt.ylabel('Residual')
    plt.grid()
    plt.show()