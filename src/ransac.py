import open3d as o3d
import numpy as np

# PCD 파일 경로 설정
pcd_path = "extrinsic_data/valid_pointclouds/0010.pcd"  # PCD 파일 경로

# 포인트 클라우드 로드
pcd = o3d.io.read_point_cloud(pcd_path)
print("Loaded PCD file:", pcd_path)

# 원본 복사 (처음 상태 저장)
original_pcd = pcd

# 파라미터 설정
max_planes = 10  # 최대 평면 개수
distance_threshold = 0.05  # 평면과 점 사이의 거리 임계값 (단위: m)
ransac_n = 3  # RANSAC에 필요한 최소 포인트 수
num_iterations = 1000  # RANSAC 반복 횟수

planes = []  # 검출된 평면 리스트
axes = []  # 평면 축 리스트
colors = [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [1, 0, 1]]  # 색상 리스트

# 평면 검출 루프
for i in range(max_planes):
    print(f"Finding plane {i+1}...")

    # RANSAC을 사용하여 평면 검출
    plane_model, inliers = pcd.segment_plane(distance_threshold, ransac_n, num_iterations)

    if len(inliers) < 100:  # 너무 작은 평면은 무시
        print("Too few points for further planes. Stopping.")
        break

    # 평면 포인트 분리
    plane = pcd.select_by_index(inliers)
    plane.paint_uniform_color(colors[i % len(colors)])  # 색상 적용
    planes.append(plane)

    # 평면에 속하지 않는 점
    remaining_pcd = pcd.select_by_index(inliers, invert=True)
    remaining_pcd.paint_uniform_color([0.5, 0.5, 0.5])  # 회색 (비평면 점)

    # 평면의 중심 및 법선 벡터 계산
    points = np.asarray(plane.points)
    center = points.mean(axis=0)  # 평면 중심
    normal = np.array(plane_model[:3])  # 평면 법선 벡터
    normal /= np.linalg.norm(normal)  # 정규화

    # 평면의 축을 시각화 (법선 벡터를 Z축으로 정렬)
    axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2)

    # 회전 행렬 생성 (법선 벡터를 Z축에 맞춤)
    z_axis = np.array([0, 0, 1])  # 기본 Z축
    R = np.eye(3)  # 기본 회전 행렬

    if not np.allclose(normal, z_axis):  # 법선이 Z축과 다르면 회전 적용
        v = np.cross(z_axis, normal)  # 회전 축 계산
        s = np.linalg.norm(v)  # 회전 크기
        c = np.dot(z_axis, normal)  # 회전 각도 (cosθ)

        skew_symmetric = np.array([
            [0, -v[2], v[1]],
            [v[2], 0, -v[0]],
            [-v[1], v[0], 0]
        ])

        R = np.eye(3) + skew_symmetric + (skew_symmetric @ skew_symmetric) * ((1 - c) / (s**2))

    # 변환 행렬 생성 (회전 + 평면 중심 이동)
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = center  # 평면의 중심 이동

    # 좌표축 변환 적용
    axis.transform(T)
    axes.append(axis)

    # 남은 포인트 클라우드 업데이트
    pcd = remaining_pcd

    # Open3D 시각화
    print("Displaying detected planes and their axes...")
    o3d.visualization.draw_geometries(planes + axes + [pcd], window_name="Multiple Planes with Axes")
