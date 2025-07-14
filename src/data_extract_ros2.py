#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, threading
import rclpy
import numpy as np
import open3d as o3d
import cv2
import sensor_msgs_py.point_cloud2 as pc2

from datetime import datetime
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import CompressedImage, Image, PointCloud2
from cv_bridge import CvBridge
from pynput import keyboard

class ImagePointCloudRecorder(Node):
    def __init__(self):
        super().__init__('image_pointcloud_recorder')

        # declare parameters
        self.declare_parameter('use_compressed_image', True)
        self.declare_parameter('image_topic', '/my_camera/pylon_ros2_camera_node/image_raw/compressed')
        self.declare_parameter('pc_topic', '/velodyne_points')
        self.declare_parameter('base_dir', '~/ROS2/calibration_ws/src/Camera-Lidar-Extrinsic-Calibration/result3')

        use_compressed = self.get_parameter('use_compressed_image').get_parameter_value().bool_value
        self.image_topic = self.get_parameter('image_topic').get_parameter_value().string_value
        self.pc_topic = self.get_parameter('pc_topic').get_parameter_value().string_value
        base_dir = os.path.expanduser(self.get_parameter('base_dir').get_parameter_value().string_value)

        # prepare directories
        self.cam_dir = os.path.join(base_dir, 'camera')
        self.ouster_dir = os.path.join(base_dir, 'ouster')
        os.makedirs(self.cam_dir, exist_ok=True)
        os.makedirs(self.ouster_dir, exist_ok=True)

        self.bridge = CvBridge()
        self.latest_image_msg = None
        self.latest_pc_msg = None

        # QoS for point cloud
        self.pc_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )

        # subscribe to image topic (compressed or raw)
        if use_compressed:
            self.get_logger().info(f"Using CompressedImage on '{self.image_topic}'")
            self.create_subscription(CompressedImage, self.image_topic, self.image_cb, 10)
        else:
            self.get_logger().info(f"Using raw Image on '{self.image_topic}'")
            self.create_subscription(Image, self.image_topic, self.image_cb, 10)

        # subscribe to point cloud topic
        self.create_subscription(PointCloud2, self.pc_topic, self.pc_cb, self.pc_qos)

        self.get_logger().info(f"📸 {self.image_topic}  ,  🟢 {self.pc_topic}")
        self.get_logger().info(f"저장 위치  image→{self.cam_dir}   pcd→{self.ouster_dir}")
        self.get_logger().info("▶ 터미널에서 's' 키를 누르면 저장합니다")

        threading.Thread(target=self.keyboard_listener, daemon=True).start()

    def image_cb(self, msg):
        # store the latest image message (compressed or raw)
        self.latest_image_msg = msg

    def pc_cb(self, msg):
        self.latest_pc_msg = msg

    def keyboard_listener(self):
        def on_press(key):
            if key == keyboard.KeyCode.from_char('s'):
                self.save_current()
        with keyboard.Listener(on_press=on_press) as l:
            l.join()

    def save_current(self):
        if self.latest_image_msg is None or self.latest_pc_msg is None:
            self.get_logger().warn("아직 두 토픽 모두 수신되지 않았습니다")
            return

        stamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        # --- Save image ---
        try:
            if isinstance(self.latest_image_msg, CompressedImage):
                img = self.bridge.compressed_imgmsg_to_cv2(self.latest_image_msg, 'bgr8')
            else:
                img = self.bridge.imgmsg_to_cv2(self.latest_image_msg, 'bgr8')

            img_path = os.path.join(self.cam_dir, f'{stamp}.png')
            cv2.imwrite(img_path, img)
            self.get_logger().info(f"✓ 이미지 저장 → {img_path}")
        except Exception as e:
            self.get_logger().error(f"이미지 저장 실패: {e}")

        # --- Save point cloud ---
        try:
            # build Nx3 float32 array
            xyz = np.array(
                [[pt[0], pt[1], pt[2]] for pt in pc2.read_points(
                    self.latest_pc_msg,
                    field_names=('x', 'y', 'z'),
                    skip_nans=True
                )],
                dtype=np.float32
            )

            if xyz.size == 0:
                self.get_logger().warn("PointCloud2 비어 있어 저장 생략")
                return

            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(xyz)
            pc_path = os.path.join(self.ouster_dir, f'{stamp}.pcd')
            o3d.io.write_point_cloud(pc_path, pcd)
            self.get_logger().info(f"✓ PCD 저장  → {pc_path}")
        except Exception as e:
            self.get_logger().error(f"PCD 저장 실패: {e}")


def main(args=None):
    rclpy.init(args=args)
    node = ImagePointCloudRecorder()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
