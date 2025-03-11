import cv2
import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster
import matplotlib.pyplot as plt

# 1. 이미지 로드
img = cv2.imread('extrinsic_data/valid_images/0001.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 2. 체스보드 코너 검출
CHECKERBOARD = (4, 5)
ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, None)

if not ret:
    print("체스보드 코너를 찾지 못했습니다.")
    exit()

corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), 
                           criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
corners = corners.reshape(-1, 2)

# 3. 원근 변환으로 보정
corners_4 = np.array([
    corners[0],           # 왼쪽 위
    corners[CHECKERBOARD[0] - 1],  # 오른쪽 위
    corners[-1],          # 오른쪽 아래
    corners[-CHECKERBOARD[0]]      # 왼쪽 아래
], dtype=np.float32)

# 격자 간격을 반영한 목표 좌표 (5x6 격자에 맞춤)
width = 600  # 더 큰 해상도로 설정
height = 500
square_size = width / 6  # 6열로 나누기
dst_pts = np.array([
    [0, 0],
    [width - 1, 0],
    [width - 1, height - 1],
    [0, height - 1]
], dtype=np.float32)
M = cv2.getPerspectiveTransform(corners_4, dst_pts)
warped = cv2.warpPerspective(img, M, (width, height))
warped_gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)

# 4. ROI 설정
x_min, x_max = 0, width
y_min, y_max = 0, height
roi_gray = warped_gray

# 5. Canny Edge Detection
edges = cv2.Canny(roi_gray, 20, 120)
cv2.imshow('Canny Edges (Warped)', edges)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 6. HoughLinesP로 직선 검출
lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=30, minLineLength=50, maxLineGap=10)
print("Hough LinesP detected:", len(lines) if lines is not None else "None")

# 7. 수평선과 수직선 분리 및 교차점 계산
horizontal_lines = []
vertical_lines = []
intersections = []

if lines is not None:
    print("Detected line angles:")
    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
        angle = angle if angle >= 0 else angle + 180
        print(f"x1: {x1}, y1: {y1}, x2: {x2}, y2: {y2}, angle: {angle:.2f}")
        # 각도 범위 확장
        if 0 <= angle <= 15 or 165 <= angle <= 180:
            horizontal_lines.append(line[0])
        elif 75 <= angle <= 105:
            vertical_lines.append(line[0])
    print("Horizontal lines:", len(horizontal_lines))
    print("Vertical lines:", len(vertical_lines))

    if len(vertical_lines) == 0 or len(horizontal_lines) == 0:
        print("Insufficient lines detected. Check angle ranges or Hough parameters.")

    for h_x1, h_y1, h_x2, h_y2 in horizontal_lines:
        for v_x1, v_y1, v_x2, v_y2 in vertical_lines:
            def line_intersection(line1, line2):
                xdiff = (line1[0] - line1[2], line2[0] - line2[2])
                ydiff = (line1[1] - line1[3], line2[1] - line2[3])
                def det(a, b):
                    return a[0] * b[1] - a[1] * b[0]
                div = det(xdiff, ydiff)
                if div == 0:
                    return None
                d = (det([line1[0], line1[1]], [line1[2], line1[3]]), det([line2[0], line2[1]], [line2[2], line2[3]]))
                x = det(d, xdiff) / div
                y = det(d, ydiff) / div
                return x, y

            intersect = line_intersection([h_x1, h_y1, h_x2, h_y2], [v_x1, v_y1, v_x2, v_y2])
            if intersect and 0 <= intersect[0] < width and 0 <= intersect[1] < height:
                intersections.append(list(intersect))

intersections = np.array(intersections)

# 8. Hierarchical Clustering으로 교차점 그룹화
if len(intersections) > 0:
    print(f"Intersections found: {len(intersections)}, shape: {intersections.shape}")
    Z = linkage(intersections, method='ward')
    t_value = 20  # 조정 가능
    clusters = fcluster(Z, t=t_value, criterion='distance')
    print(f"Clusters shape: {clusters.shape}")

    unique_clusters = np.unique(clusters)
    grid_points = []
    for cluster_id in unique_clusters:
        cluster_mask = (clusters == cluster_id)
        cluster_points = intersections[cluster_mask]
        if len(cluster_points) > 0:
            mean_point = np.mean(cluster_points, axis=0)
            # 격자 내에 있는 점만 유지
            if 0 <= mean_point[0] < width and 0 <= mean_point[1] < height:
                grid_points.append(mean_point)
    grid_points = np.array(grid_points)
    print(f"Grid points shape: {grid_points.shape}")

    if len(grid_points) >= 30:
        # y 좌표로 정렬 (5개 행)
        grid_points = grid_points[np.argsort(grid_points[:, 1])]
        # 상위 30개 점만 선택 (5행 x 6열)
        grid_points = grid_points[:30]
        rows = [grid_points[i:i+6] for i in range(0, 30, 6)]
        for row in rows:
            row[:] = row[np.argsort(row[:, 0])]
        border = np.array([rows[0][0], rows[0][-1], rows[-1][-1], rows[-1][0]])

        # 전체 grid_points를 한 번에 역매핑
        grid_points_3d = grid_points.reshape(-1, 1, 2).astype(np.float32)
        grid_points_orig = cv2.perspectiveTransform(grid_points_3d, np.linalg.inv(M)).reshape(-1, 2)
        border_3d = border.reshape(-1, 1, 2).astype(np.float32)
        border_orig = cv2.perspectiveTransform(border_3d, np.linalg.inv(M)).reshape(-1, 2)

        for pt in grid_points_orig:
            cv2.circle(img, (int(pt[0]), int(pt[1])), 3, (0, 255, 0), -1)
        cv2.polylines(img, [border_orig.astype(int)], isClosed=True, color=(0, 0, 255), thickness=2)
        cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)

        cv2.imshow('4x5 Chessboard Detection', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        plt.scatter(intersections[:, 0], intersections[:, 1], c=clusters, cmap='viridis')
        plt.scatter(grid_points[:, 0], grid_points[:, 1], c='red', marker='x')
        plt.title('4x5 Chessboard Grid Points')
        plt.show()
    else:
        print(f"Insufficient intersections found: {len(grid_points)} (expected ~30)")
else:
    print("No intersections found.")