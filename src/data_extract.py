#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
image_pointcloud_recorder_ros1.py
's' 키 입력 시 Image·PointCloud2를 개별 폴더(camera / ouster)에 저장
"""

import os, threading
from datetime import datetime

import rospy
from sensor_msgs.msg import CompressedImage, Image, PointCloud2
from cv_bridge import CvBridge
import numpy as np
import open3d as o3d
import sensor_msgs.point_cloud2 as pc2
from pynput import keyboard
import cv2


class ImagePointCloudRecorder:
    def __init__(self):
        # ───────── 파라미터 ─────────
        self.image_topic = rospy.get_param('~image_topic', '/blackfly/image_color/compressed')
        self.pc_topic    = rospy.get_param('~pc_topic',    '/ouster/points')
        base_dir         = os.path.expanduser(
            rospy.get_param('~base_dir', '/home/myungw00/ROS/gm/Code/광양센서팩캘/projection_test/extrinsic_data2'))

        # 하위 폴더(camera / ouster) 준비
        self.cam_dir    = os.path.join(base_dir, 'camera')
        self.ouster_dir = os.path.join(base_dir, 'ouster')
        os.makedirs(self.cam_dir,    exist_ok=True)
        os.makedirs(self.ouster_dir, exist_ok=True)

        # ───────── 구독자 ─────────
        self.bridge = CvBridge()
        self.latest_image_msg = None
        self.latest_pc_msg    = None
        rospy.Subscriber(self.image_topic, CompressedImage, self.image_cb, queue_size=5)
        rospy.Subscriber(self.pc_topic,    PointCloud2, self.pc_cb, queue_size=5)

        rospy.loginfo("📸 %s  ,  🟢 %s", self.image_topic, self.pc_topic)
        rospy.loginfo("저장 위치  image→%s   pcd→%s", self.cam_dir, self.ouster_dir)
        rospy.loginfo("▶ 터미널에서 's' 키를 누르면 저장합니다")

        threading.Thread(target=self.keyboard_listener, daemon=True).start()

    # ───────── 콜백 ─────────
    def image_cb(self, msg):  self.latest_image_msg = msg
    def pc_cb(self, msg):     self.latest_pc_msg    = msg

    # ───────── 키 입력 ─────────
    def keyboard_listener(self):
        def on_press(key):
            if key == keyboard.KeyCode.from_char('s'):
                self.save_current()
        with keyboard.Listener(on_press=on_press) as l: l.join()

    # ───────── 저장 루틴 ─────────
    def save_current(self):
        if self.latest_image_msg is None or self.latest_pc_msg is None:
            rospy.logwarn("아직 두 토픽 모두 수신되지 않았습니다")
            return

        stamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        # 이미지 저장
        try:
            img = self.bridge.compressed_imgmsg_to_cv2(self.latest_image_msg, 'bgr8')
            img_path = os.path.join(self.cam_dir, f'{stamp}.png')
            cv2.imwrite(img_path, img)
            rospy.loginfo("✓ 이미지 저장 → %s", img_path)
        except Exception as e:
            rospy.logerr("이미지 저장 실패: %s", e)

        # 포인트클라우드 저장
        try:
            xyz = np.asarray(
                list(pc2.read_points(self.latest_pc_msg,
                                     field_names=('x','y','z'),
                                     skip_nans=True)),
                dtype=np.float32)
            if xyz.size == 0:
                rospy.logwarn("PointCloud2 비어 있어 저장 생략")
                return
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(xyz)
            pc_path = os.path.join(self.ouster_dir, f'{stamp}.pcd')
            o3d.io.write_point_cloud(pc_path, pcd)
            rospy.loginfo("✓ PCD 저장  → %s", pc_path)
        except Exception as e:
            rospy.logerr("PCD 저장 실패: %s", e)


if __name__ == '__main__':
    rospy.init_node('image_pointcloud_recorder')
    ImagePointCloudRecorder()
    rospy.spin()
