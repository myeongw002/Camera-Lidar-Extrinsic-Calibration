import cv2
import numpy as np

# ========================== 📌 1. 카메라 내부 파라미터 로드 ==========================

intrinsic_path = "24252427/intrinsic.csv"

# 카메라 내부 파라미터 로드
intrinsic_param = np.loadtxt(intrinsic_path, delimiter=',', usecols=range(9))
intrinsic = np.array([[intrinsic_param[0], intrinsic_param[1], intrinsic_param[2]],
                      [0.0, intrinsic_param[3], intrinsic_param[4]],
                      [0.0, 0.0, 1.0]])
distortion = np.array(intrinsic_param[5:])

# ========================== 📌 2. 체스보드 설정 ==========================

# 체스보드 크기 (내부 코너 개수: 행 × 열)
CHECKERBOARD = (4, 5)

# 체스보드 한 칸 크기 (미터 단위)
square_size = 0.14  # 14cm

# **체스보드 패딩 크기 (미터 단위)**
padding_x = 0.0105  # 가로 방향 패딩 (10cm)
padding_y = 0.021  # 세로 방향 패딩 (20cm)

# 📌 내부 코너 3D 좌표 생성 (Z=0 평면)
objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2) * square_size
print("📌 3D Object Points:\n", objp)
# **체스보드 전체 크기 계산**
board_width = CHECKERBOARD[0] * square_size
board_height = CHECKERBOARD[1] * square_size

# 📌 패딩 포함한 체스보드 네 모서리 추가 (패딩 적용)
board_corners = np.array([
    [objp[0][0] - square_size - padding_x, objp[0][1] - square_size - padding_y, 0],   # 좌측 상단
    [objp[3][0] + square_size + padding_x, objp[3][1] - square_size - padding_y, 0],   # 우측 상단
    [objp[19][0]  + square_size + padding_x, objp[19][1] + square_size + padding_y, 0], # 우측 하단
    [objp[16][0] - square_size - padding_x, objp[16][1] + square_size + padding_y, 0], # 좌측 하단
], dtype=np.float32)
print("📌 Board Corners with Padding:\n", board_corners)
# `objp`에 패딩 모서리 추가 (행렬 결합)
objp_with_padding = np.vstack((objp, board_corners))

# ========================== 📌 3. SolvePnP를 사용한 3D 위치 추정 ==========================

# 📌 PnP를 수행할 테스트 이미지 선택
test_img_path = "extrinsic_data/valid_images/0001.jpg"
test_img = cv2.imread(test_img_path)
test_gray = cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)

# 📌 체스보드 코너 검출
ret, corners = cv2.findChessboardCorners(test_gray, CHECKERBOARD, None)

if ret:
    corners = corners.reshape(-1, 2)  # 2D 좌표로 변환

    if corners[0][1] > corners[-1][1]:
        corners = np.flip(corners, axis=0)

    corners = corners.reshape(-1, 1, 2).astype(np.float32)

    refined_corners = cv2.cornerSubPix(
        test_gray, corners, (11, 11), (-1, -1),
        criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    )

    # 📌 SolvePnP를 사용하여 회전 및 이동 벡터 추정
    success, rvec, tvec = cv2.solvePnP(objp, refined_corners, intrinsic, distortion)

    if success:
        print("📌 Estimated Rotation Vector:\n", rvec)
        print("📌 Estimated Translation Vector:\n", tvec)

        # ========================== 📌 4. 재투영 (Reprojection) 검증 ==========================

        # 📌 내부 체스보드 코너 재투영
        reprojected_points, _ = cv2.projectPoints(objp, rvec, tvec, intrinsic, distortion)

        # 📌 체스보드 외곽 모서리 재투영
        reprojected_corners, _ = cv2.projectPoints(board_corners, rvec, tvec, intrinsic, distortion)

        # 📌 원본 이미지에 재투영된 점을 표시
        for pt in reprojected_points.squeeze():
            cv2.circle(test_img, (int(pt[0]), int(pt[1])), 5, (255, 0, 0), -1)  # 파란색 점

        # 📌 체스보드 외곽 모서리 선으로 연결
        reprojected_corners = reprojected_corners.squeeze().astype(int)
        cv2.polylines(test_img, [reprojected_corners], isClosed=True, color=(0, 255, 0), thickness=2)

        # 📌 결과 출력
        cv2.imshow("Reprojected Chessboard with Padding", test_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    else:
        print("❌ SolvePnP 실패")
else:
    print("❌ 체스보드 코너를 찾을 수 없음")
