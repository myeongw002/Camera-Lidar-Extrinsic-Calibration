## Camera - Lidar Extrinsic Calibration based on IOU Optimization


## How to use

1. rosbag 녹화 
2. data_extract.py → rosbag에서 데이터 추출
3. [intrinsic_numpy.py](http://intrinsic.py) → 카메라 intrinsic 계산
4. check_validation.py → 데이터 유효성 검사
5. [extrinsic.py](http://extrinsic.py) → extrinsic parameter 계산


## Images

Original image
![Original image](docs/1.jpg)

Unoptimized IOU image
![Unoptimized IOU image](docs/2.jpg)

Optimized IOU image
![Optimized IOU image](docs/3.jpg)

Lidar projection image
![Lidar projection image](docs/4.jpg)