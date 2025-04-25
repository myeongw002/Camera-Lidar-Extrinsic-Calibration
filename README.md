# Camera - Lidar Extrinsic Calibration based on IOU Optimization

## Explanation
https://www.notion.so/Camera-Lidar-Extrinsic-calibration-1b384728f3f1809596c3ce1e6ba3dbe1?pvs=4
## How to use

1. rosbag 녹화 
2. data_extract.py → rosbag에서 데이터 추출
3. intrinsic_numpy.py → 카메라 intrinsic 계산
4. check_validation.py → 데이터 유효성 검사
5. extrinsic.py → extrinsic parameter 계산


## Images

Original image
![Original image](docs/1.jpg)

Unoptimized IOU image
![Unoptimized IOU image](docs/2.jpg)

Optimized IOU image
![Optimized IOU image](docs/3.jpg)

Lidar projection image
![Lidar projection image](docs/4.jpg)


## References

1. **Jiunn-Kai Huang, Jessy W. Grizzle**, “Improvements to Target-Based 3D LiDAR to Camera Calibration,” _IEEE Access_, vol. 8, pp. 134101–134110, 2020.  
   🔗 [DOI](https://doi.org/10.1109/ACCESS.2020.3010734)  
   🔗 [arXiv:1910.03126](https://arxiv.org/abs/1910.03126)