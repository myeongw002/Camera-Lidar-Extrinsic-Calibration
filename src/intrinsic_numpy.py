import numpy as np
import cv2
import glob
from scipy.optimize import least_squares
import os 

# 정규화 함수
def normalize_points(points):
    """이미지 좌표를 정규화하여 수치적 안정성을 높임"""
    mean = np.mean(points, axis=0)
    std = np.std(points, axis=0)
    T = np.array([[1/std[0], 0, -mean[0]/std[0]],
                  [0, 1/std[1], -mean[1]/std[1]],
                  [0, 0, 1]])
    points_h = np.hstack((points, np.ones((points.shape[0], 1))))
    points_norm = (T @ points_h.T).T[:, :2]
    return points_norm, T

# Homography 계산 (정규화 적용)
def compute_homography(world_points, image_points):
    """정규화된 좌표를 사용하여 Homography를 계산"""
    image_points_norm, T = normalize_points(image_points)
    print(image_points_norm)
    world_points_norm, Tw = normalize_points(world_points[:, :2])
    
    A = []
    for (X, Y), (x, y) in zip(world_points_norm, image_points_norm):
        A.append([-X, -Y, -1, 0, 0, 0, X*x, Y*x, x])
        A.append([0, 0, 0, -X, -Y, -1, X*y, Y*y, y])
    A = np.array(A)
    _, _, Vt = np.linalg.svd(A)
    H_norm = Vt[-1].reshape(3, 3)
    H = np.linalg.inv(T) @ H_norm @ Tw
    return H / H[2, 2]

# 내재 파라미터 추정 (Zhang's method)
def compute_intrinsic(homographies):
    """Homography를 사용하여 초기 내재 파라미터(K)를 추정"""
    def v_ij(H, i, j):
        return np.array([
            H[0, i]*H[0, j],
            H[0, i]*H[1, j] + H[1, i]*H[0, j],
            H[1, i]*H[1, j],
            H[2, i]*H[0, j] + H[0, i]*H[2, j],
            H[2, i]*H[1, j] + H[1, i]*H[2, j],
            H[2, i]*H[2, j]
        ])

    V = np.vstack([v_ij(H, 0, 1) for H in homographies] + 
                  [v_ij(H, 0, 0) - v_ij(H, 1, 1) for H in homographies])
    
    _, _, Vt = np.linalg.svd(V)
    b = Vt[-1]
    B = np.array([
        [b[0], b[1], b[3]],
        [b[1], b[2], b[4]],
        [b[3], b[4], b[5]]
    ])
    
    v0 = (B[0,1]*B[0,2] - B[0,0]*B[1,2]) / (B[0,0]*B[1,1] - B[0,1]**2)
    lam = B[2,2] - (B[0,2]**2 + v0*(B[0,1]*B[0,2] - B[0,0]*B[1,2])) / B[0,0]
    alpha = np.sqrt(lam / B[0,0]) 
    beta  = np.sqrt(lam * B[0,0] / (B[0,0]*B[1,1] - B[0,1]**2))
    gamma = -B[0,1]*alpha**2 * beta / lam
    u0 = gamma*v0 / beta - B[0,2]*alpha**2 / lam
    
    K = np.array([
        [alpha, gamma, u0],
        [0,     beta,  v0],
        [0,     0,     1 ]
    ])
    return K

# 외재 파라미터 추정
def compute_extrinsics(homographies, K):
    """내재 파라미터를 이용해 초기 외재 파라미터(R, t)를 계산"""
    K_inv = np.linalg.inv(K)
    extrinsics = []
    for H in homographies:
        h1, h2, h3 = H[:, 0], H[:, 1], H[:, 2]
        lam = 1.0 / np.linalg.norm(K_inv @ h1)
        r1 = lam * (K_inv @ h1)
        r2 = lam * (K_inv @ h2)
        r3 = np.cross(r1, r2)
        t = lam * (K_inv @ h3)
        R = np.column_stack((r1, r2, r3))
        extrinsics.append((R, t))
    return extrinsics

# 재투영 오차 함수 (동시 최적화용)
def reprojection_error(params, objpoints, imgpoints, num_boards):
    """내재 파라미터, 왜곡 계수, 외재 파라미터를 사용해 재투영 오차 계산"""
    fx, fy, cx, cy = params[:4]
    dist_coeffs = params[4:9]  # k1, k2, p1, p2, k3
    rvecs = params[9:9 + 3*num_boards].reshape(num_boards, 3)
    tvecs = params[9 + 3*num_boards:].reshape(num_boards, 3)
    
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
    residuals = []
    for i, (objp, imgp) in enumerate(zip(objpoints, imgpoints)):
        R, _ = cv2.Rodrigues(rvecs[i])  # 회전 벡터를 회전 행렬로 변환
        t = tvecs[i]
        pts_cam = objp @ R.T + t
        x_cam = pts_cam[:, 0] / pts_cam[:, 2]
        y_cam = pts_cam[:, 1] / pts_cam[:, 2]
        r2 = x_cam**2 + y_cam**2
        
        k1, k2, p1, p2, k3 = dist_coeffs
        radial = 1 + k1*r2 + k2*r2**2 + k3*r2**3
        x_dist = x_cam*radial + 2*p1*x_cam*y_cam + p2*(r2 + 2*x_cam**2)
        y_dist = y_cam*radial + p1*(r2 + 2*y_cam**2) + 2*p2*x_cam*y_cam
        
        u = fx*x_dist + cx
        v = fy*y_dist + cy
        
        residuals.append(u - imgp[:, 0])
        residuals.append(v - imgp[:, 1])
    
    return np.concatenate(residuals)

if __name__ == "__main__":
    # 체스보드 패턴 설정
    nx, ny = 5, 7  # 체스보드 코너 개수
    square_size = 0.095  # 체스보드 한 칸의 크기 (단위: m)
    objp = np.zeros((nx * ny, 3), np.float32)
    objp[:, :2] = np.mgrid[0:nx, 0:ny].T.reshape(-1, 2) * square_size

    objpoints, imgpoints = [], []
    images = sorted(glob.glob('extrinsic_data2/camera/*.png'))  # 이미지 파일 경로
    
    # 체스보드 코너 검출
    for fname in images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)
        if ret:
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1),
                                        (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
            objpoints.append(objp)
            imgpoints.append(corners2.reshape(-1, 2))

    # Homography 계산 (정규화 적용)
    homographies = [compute_homography(o, i) for o, i in zip(objpoints, imgpoints)]

    # 내재 파라미터 초기 추정
    K_init = compute_intrinsic(homographies)
    print("[Initial K]\n", K_init)

    # 외재 파라미터 초기 추정
    extrinsics_init = compute_extrinsics(homographies, K_init)
    rvecs_init = [cv2.Rodrigues(R)[0].ravel() for R, _ in extrinsics_init]
    tvecs_init = [t.ravel() for _, t in extrinsics_init]

    # 초기 파라미터 설정
    num_boards = len(objpoints)
    params_init = np.hstack([
        K_init[0,0], K_init[1,1], K_init[0,2], K_init[1,2],  # fx, fy, cx, cy
        np.zeros(5),  # k1, k2, p1, p2, k3
        np.vstack(rvecs_init).ravel(),
        np.vstack(tvecs_init).ravel()
    ])

    # 동시 최적화
    result = least_squares(
        reprojection_error,
        params_init,
        args=(objpoints, imgpoints, num_boards),
        method='lm'  # Levenberg-Marquardt 알고리즘 사용
    )

    # 최적화된 파라미터 추출
    fx_opt, fy_opt, cx_opt, cy_opt = result.x[:4]
    dist_coeffs_opt = result.x[4:9]
    rvecs_opt = result.x[9:9 + 3*num_boards].reshape(num_boards, 3)
    tvecs_opt = result.x[9 + 3*num_boards:].reshape(num_boards, 3)

    # -------- [최적화 후] -------------------------------------------------
    K_opt = np.array([[fx_opt, 0.0, cx_opt],
                    [0.0,   fy_opt, cy_opt],
                    [0.0,     0.0,   1.0 ]])

    # 저장할 1×9 벡터 구성
    params_to_save = np.array([
        fx_opt,          # 0
        0.0,             # 1: gamma(=skew), 일반적으로 0
        cx_opt,          # 2
        fy_opt,          # 3
        cy_opt,          # 4
        dist_coeffs_opt[0],  # 5: k1
        dist_coeffs_opt[1],  # 6: k2
        dist_coeffs_opt[2],  # 7: p1
        dist_coeffs_opt[3],   # 8: p2
        dist_coeffs_opt[4]
    ])
    print("Intrinsic")

    # -------- [저장 경로 설정] --------------------------------------------
    # 이미지가 들어 있는 폴더를 기준으로 intrinsic.csv 생성
    image_dir = os.path.dirname(images[0])      # 예: extrinsic_data/camera
    intrinsic_path = os.path.join(image_dir, "intrinsic.csv")

    # -------- [파일 저장] --------------------------------------------------
    np.savetxt(intrinsic_path,
            params_to_save.reshape(1, -1),
            delimiter=',',
            fmt='%.10f')   # 소수점 10자리; 필요 시 조정
    print(f"[Saved] intrinsic parameters → {intrinsic_path}")
