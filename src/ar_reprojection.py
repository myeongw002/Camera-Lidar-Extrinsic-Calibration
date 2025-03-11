import cv2
import numpy as np

intrinsic_path = "24252427/intrinsic.csv"

# 카메라 내부 파라미터 로드
intrinsic_param = np.loadtxt(intrinsic_path, delimiter=',', usecols=range(9))
intrinsic = np.array([[intrinsic_param[0], intrinsic_param[1], intrinsic_param[2]],
                      [0.0, intrinsic_param[3], intrinsic_param[4]],
                      [0.0, 0.0, 1.0]])
distortion = np.array(intrinsic_param[5:])

# 📌 ArUco 마커 사전 설정
aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_250)
parameters = cv2.aruco.DetectorParameters_create()

# 📌 마커 실제 크기 (미터 단위)
marker_size = 0.17  # 10cm

# 📌 3D 마커 코너 좌표 설정 (왼쪽 상단부터 시계 방향)
object_points = np.array([
    [-marker_size / 2, marker_size / 2, 0],  # 왼쪽 상단
    [marker_size / 2, marker_size / 2, 0],   # 오른쪽 상단
    [marker_size / 2, -marker_size / 2, 0],  # 오른쪽 하단
    [-marker_size / 2, -marker_size / 2, 0]  # 왼쪽 하단
], dtype=np.float32)

# 📌 마커의 중심 3D 좌표
marker_center_3D = np.array([[0, 0, 0]], dtype=np.float32)  # (X=0, Y=0, Z=0)

# 📌 이미지 로드 및 ArUco 마커 검출
image_path = "extrinsic_data/valid_images/0001.jpg"  # 이미지 경로 수정
image = cv2.imread(image_path)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# ArUco 마커 검출
corners, ids, rejected = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=parameters)

# 📌 마커가 검출되었을 경우 3D 위치 추정 및 재투영 수행
if ids is not None:
    cv2.aruco.drawDetectedMarkers(image, corners, ids)  # 검출된 마커 표시
    for i in range(len(ids)):
        # SolvePnP 사용하여 마커의 3D 위치 및 회전 추정
        ret, rvec, tvec = cv2.solvePnP(object_points, corners[i], intrinsic, distortion)

        if ret:
            print(f"Marker ID {ids[i][0]} - Rotation Vector:\n", rvec)
            print(f"Marker ID {ids[i][0]} - Translation Vector:\n", tvec)

            # 3D 마커 중심을 2D 이미지 좌표로 투영
            projected_center, _ = cv2.projectPoints(marker_center_3D, rvec, tvec, intrinsic, distortion)

            # 투영된 마커 중심 좌표 가져오기
            x, y = int(projected_center[0][0][0]), int(projected_center[0][0][1])
            print(f"Marker {ids[i][0]} projected center: ({x}, {y})")

            # 이미지에 마커 중심을 초록색 점으로 표시
            cv2.circle(image, (x, y), 5, (255, 255, 255), -1)  # 초록색 원 그리기
            cv2.putText(image, f"ID:{ids[i][0]}", (x + 10, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # 3D 좌표축 시각화
            cv2.drawFrameAxes(image, intrinsic, distortion, rvec, tvec, marker_size / 2)

else:
    print("No ArUco markers detected.")

# 📌 결과 출력
cv2.imshow("ArUco Marker Reprojection", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
