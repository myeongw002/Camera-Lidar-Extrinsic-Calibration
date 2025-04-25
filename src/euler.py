import numpy as np
from scipy.spatial.transform import Rotation

# ──────────────────────────────────────────────────────────────────────
# 1.  텍스트 파일 → 4×4 행렬
#    - 각 행은 콤마로 구분된 4개의 실수
#    - 예:  iou_optimized_transform.txt
# ──────────────────────────────────────────────────────────────────────
T = np.loadtxt("iou_optimized_transform1.txt", delimiter=",")     # shape = (4, 4)
assert T.shape == (4, 4), "파일 형식이 4×4 행렬이어야 합니다."

# ──────────────────────────────────────────────────────────────────────
# 2.  회전 행렬 → 오일러 각 (XYZ; roll‑pitch‑yaw)
# ──────────────────────────────────────────────────────────────────────
R = T[:3, :3]                                # 회전 부분 추출
r = Rotation.from_matrix(R)                  # SciPy 객체 생성
roll, pitch, yaw = r.as_euler("xyz", degrees=True)   # deg 단위

print(f"roll  (X) : {roll: .3f}°")
print(f"pitch (Y) : {pitch:.3f}°")
print(f"yaw   (Z) : {yaw: .3f}°")

# 필요하다면 rad 단위로도 얻을 수 있습니다.
roll, pitch, yaw = r.as_euler("xyz", degrees=False)

print(f"roll  (X) : {roll: .3f}°")
print(f"pitch (Y) : {pitch:.3f}°")
print(f"yaw   (Z) : {yaw: .3f}°")
