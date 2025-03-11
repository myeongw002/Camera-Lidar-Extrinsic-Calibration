import numpy as np
import open3d as o3d
import cv2
from copy import deepcopy
from scipy.spatial import ConvexHull
from sklearn.decomposition import PCA
from scipy.spatial import distance
import itertools


def perform_icp(source, target, threshold, init_transform, visualize):
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
    if visualize:
        o3d.visualization.draw_geometries([source_cp, target], window_name="ICP Result")
    # 변환 행렬 반환
    # print("Transformation matrix:\n", reg_icp.transformation)
    return reg_icp.transformation

import numpy as np
import cv2

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
    if not pcd.has_points():
        return img, []

    # LiDAR 포인트 변환
    pcd_points = np.asarray(pcd.points)
    pcd_homogeneous = np.hstack((pcd_points, np.ones((pcd_points.shape[0], 1))))  # 3D → 4D
    transformed_pcd = (transform @ pcd_homogeneous.T).T[:, :3]  # 변환 후 3D 좌표

    # 카메라 앞쪽에 있는 점만 필터링
    valid_mask = transformed_pcd[:, 2] > 0
    transformed_pcd = transformed_pcd[valid_mask]
    if transformed_pcd.shape[0] == 0:
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

    # 이미지 범위 내 포인트 필터링
    h, w = img.shape[:2]
    valid_mask = (projected_points[:, 0] >= 0) & (projected_points[:, 0] < w) & \
                 (projected_points[:, 1] >= 0) & (projected_points[:, 1] < h)
    valid_points = projected_points[valid_mask].astype(int)
    valid_colors = colors[valid_mask]

    # 이미지에 포인트 투영 (점 크기 조절)
    for (x, y), c in zip(valid_points, valid_colors):
        cv2.circle(img, (x, y), point_size, tuple(map(int, c.squeeze())), -1)  # -1: 원을 채움

    return img, valid_points


def draw_contour(img, corners, rvec, tvec, intrinsic, distortion):
    # 체커보드 코너를 2D 이미지 평면으로 투영
    projected_corners, _ = cv2.projectPoints(corners, rvec, tvec, intrinsic, distortion)
    projected_corners = projected_corners.reshape(-1, 2)  # (N, 1, 2) -> (N, 2)

    # 이미지에 체커보드 코너 그리기
    for pt in projected_corners:
        cv2.circle(img, (int(pt[0]), int(pt[1])), 5, (0, 255, 0), -1)  # 초록색 점

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

    return filtered_pcd

def find_largest_quadrilateral(points):
    """
    Convex Hull에서 4개의 점을 선택하여 가장 넓은 사각형을 찾는 함수
    Args:
        points (numpy array): (N, 2) 형태의 Convex Hull 점들
    Returns:
        best_quad (numpy array): 가장 넓은 4개의 점 (4, 2) 형태
    """
    max_area = 0
    best_quad = None
    
    # Convex Hull 점들 중 4개를 조합하여 모든 경우 확인
    for quad in itertools.combinations(points, 4):
        quad = np.array(quad)  # 리스트를 numpy array로 변환
        
        # Convex Hull을 이용해 넓이 계산
        hull = cv2.convexHull(quad)
        area = cv2.contourArea(hull)

        if area > max_area:
            max_area = area
            best_quad = quad
    
    return best_quad  # (4, 2) 형태의 좌표 반환


def nearest_corner(quad_pts, corner_pts):
    """
    가장 넓은 Convex Hull 4점과 체스보드 코너점 중 가장 가까운 점 찾기
    Args:
        quad_pts: (4,2) 형태의 Convex Hull 점들
        corner_pts: (N,2) 형태의 체스보드 코너점들
    Returns:
        matched_pts: quad_pts에 대해 대응하는 corner_pts
    """
    matched_pts = []
    
    for quad_pt in quad_pts:
        dists = np.linalg.norm(corner_pts - quad_pt, axis=1)  # 유클리드 거리 계산
        nearest_idx = np.argmin(dists)  # 가장 가까운 코너 인덱스 찾기
        matched_pts.append(corner_pts[nearest_idx])  # 대응 코너 저장
    
    return np.array(matched_pts)  # (4,2) 형태

def reprojection_error(params, pcd_list, corner_list, intrinsic, distortion, image_T_list):
    residuals = []
    for pcd_pts, corner_pts, image_T in zip(pcd_list, corner_list, image_T_list):
        # params: 6차원 extrinsic 파라미터 (회전, 이동)
        rvec = params[:3].reshape(3,1)
        tvec = params[3:6].reshape(3,1)

        image_R = image_T[:3,:3]
        image_rvec = cv2.Rodrigues(image_R)[0]
        image_tvec = image_T[:3,3].reshape(3,1)
        
        # LiDAR 포인트 클라우드 투영
        pcd_np = np.asarray(pcd_pts.points)
        projected_pcd, _ = cv2.projectPoints(pcd_np, rvec, tvec, intrinsic, distortion)
        projected_pcd = projected_pcd.reshape(-1,2)  # (N,1,2) → (N,2)
        
        # 체스보드 코너 투영
        projected_corner, _ = cv2.projectPoints(corner_pts, image_rvec, image_tvec, intrinsic, distortion)
        projected_corner = projected_corner.reshape(-1,2)

        # LiDAR Convex Hull & 체스보드 가장 넓은 4점 찾기
        hull = cv2.convexHull(projected_pcd.astype(np.float32)).reshape(-1,2)
        quad_pts = find_largest_quadrilateral(hull)

        # Convex Hull의 4개 점과 가장 가까운 체스보드 코너 찾기
        matched_pts = nearest_corner(quad_pts, projected_corner)

        # Residuals 계산 (오차 = 대응점 간 거리)
        distances = np.linalg.norm(matched_pts - quad_pts, axis=1)
        residuals.extend(distances)
    
    return np.array(residuals)  # 최적화할 오차 배열

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
    return np.array(residuals)

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
    cv2.polylines(img, [contour_points], isClosed=True, color=(255, 0, 0), thickness=2)  # 파란색
    return img

def point_to_line_dist(pt, line):
    """
    점 `pt`와 선분 `line`(시작점, 끝점) 사이의 최소 거리 계산

    Args:
        pt (numpy array): (2,) 형태의 점 좌표 (x, y)
        line (tuple): ((x1, y1), (x2, y2)) 형태의 선분 좌표

    Returns:
        float: 점과 선분 사이의 최소 거리
    """
    start, end = line
    line_vec = end - start
    point_vec = pt - start
    line_len = np.linalg.norm(line_vec)

    # 투영된 점의 위치 파악
    proj_scalar = np.dot(point_vec, line_vec) / (line_len**2 + 1e-6)
    proj_scalar = np.clip(proj_scalar, 0, 1)  # 선분 범위 내에 위치하도록 제한

    closest_pt = start + proj_scalar * line_vec  # 선분 상 가장 가까운 점
    return np.linalg.norm(pt - closest_pt)


def compute_edge_based_error(params, pcd_list, corner_list, intrinsic, distortion, image_T_list):
    """
    LiDAR 포인트 클라우드의 Convex Hull과 체스보드 Edge 사이의 거리를 기반으로 Reprojection Error 계산.

    Args:
        params: 6차원 extrinsic 파라미터 (회전, 이동)
        pcd_list: 여러 개의 LiDAR 포인트 클라우드 리스트
        corner_list: 여러 개의 체스보드 모서리 리스트
        intrinsic: 카메라 내부 파라미터 (3x3)
        distortion: 왜곡 계수
        image_T_list: 이미지에서 추출한 체스보드 변환 행렬 리스트

    Returns:
        residuals (numpy array): 일정한 크기를 가진 residual 배열
    """
    residuals = []

    for pcd_pts, corner_pts, image_T in zip(pcd_list, corner_list, image_T_list):
        rvec = params[:3].reshape(3, 1)
        tvec = params[3:6].reshape(3, 1)

        image_R = image_T[:3, :3]
        image_rvec = cv2.Rodrigues(image_R)[0]
        image_tvec = image_T[:3, 3].reshape(3, 1)

        # **1. 체스보드의 모서리 Edge 생성 (2D 변환)**
        projected_corners, _ = cv2.projectPoints(corner_pts, rvec, tvec, intrinsic, distortion)
        projected_corners = projected_corners.reshape(-1, 2)  # (4, 2) 형태

        # 체스보드 경계선 정의 (선분 4개)
        chessboard_edges = np.array([
            [projected_corners[0], projected_corners[1]],
            [projected_corners[1], projected_corners[3]],
            [projected_corners[3], projected_corners[2]],
            [projected_corners[2], projected_corners[0]]
        ])  # (4, 2, 2) 형태

        # **2. LiDAR 포인트 Convex Hull 검출**
        pcd_np = np.asarray(pcd_pts.points)
        lidar_2d, _ = cv2.projectPoints(pcd_np, rvec, tvec, intrinsic, distortion)
        lidar_2d = lidar_2d.reshape(-1, 2)  # (N, 2)

        if lidar_2d.shape[0] < 4:
            print("Warning: Not enough LiDAR points after projection.")
            continue

        # Convex Hull 계산 (2D)
        hull_indices = cv2.convexHull(lidar_2d.astype(np.float32), returnPoints=False)
        lidar_hull_pts = lidar_2d[hull_indices.squeeze()]  # Convex Hull 점들만 가져오기

        # **3. Residual 계산 (Hull 점과 Edge 사이의 최소 거리)**
        min_residuals = np.zeros(4)  # 선분 4개에 대한 최소 residual을 저장

        for i, edge in enumerate(chessboard_edges):
            min_residuals[i] = np.min([point_to_line_dist(hull_pt, edge) for hull_pt in lidar_hull_pts])

        residuals.append(min_residuals)  # (4,) 크기의 residual 추가

    # **4. Residual을 (N, 4) → (4N,)으로 평탄화하여 일정한 크기 유지**
    return np.concatenate(residuals) if residuals else np.zeros(4 * len(pcd_list))

