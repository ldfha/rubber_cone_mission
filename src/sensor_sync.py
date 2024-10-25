#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import Imu, NavSatFix, PointCloud2, Image
from message_filters import ApproximateTimeSynchronizer, Subscriber
from cv_bridge import CvBridge

# 전역 변수 초기화
bridge = CvBridge()

def sync_callback(lidar_data, camera1_data, camera2_data, camera3_data, camera4_data, camera5_data):
            # LiDAR 데이터 퍼블리시
    synced_lidar_pub.publish(lidar_data)

            # 카메라 데이터 퍼블리시
    synced_camera1_pub.publish(camera1_data)
    synced_camera2_pub.publish(camera2_data)
    synced_camera3_pub.publish(camera3_data)
    synced_camera4_pub.publish(camera4_data)
    synced_camera5_pub.publish(camera5_data)
    


def main():
    rospy.init_node('sensor_sync_node')

    # ---- 구독자 설정 ----
    lidar_sub = Subscriber('/velodyne_points', PointCloud2)
    camera1_sub = Subscriber('/camera1/usb_cam_1/image_raw', Image)
    camera2_sub = Subscriber('/camera2/usb_cam_2/image_raw', Image)
    camera3_sub = Subscriber('/camera3/usb_cam_3/image_raw', Image)
    camera4_sub = Subscriber('/camera4/usb_cam_4/image_raw', Image)
    camera5_sub = Subscriber('/camera5/usb_cam_5/image_raw', Image)

    # ---- 퍼블리셔 설정 ----
    global synced_lidar_pub
    global synced_camera1_pub, synced_camera2_pub, synced_camera3_pub, synced_camera4_pub, synced_camera5_pub

    synced_lidar_pub = rospy.Publisher('/synced/velodyne_points', PointCloud2, queue_size=1)
    synced_camera1_pub = rospy.Publisher('/synced/camera1/usb_cam_1/image_raw', Image, queue_size=1)
    synced_camera2_pub = rospy.Publisher('/synced/camera2/usb_cam_2/image_raw', Image, queue_size=1)
    synced_camera3_pub = rospy.Publisher('/synced/camera3/usb_cam_3/image_raw', Image, queue_size=1)
    synced_camera4_pub = rospy.Publisher('/synced/camera4/usb_cam_4/image_raw', Image, queue_size=1)
    synced_camera5_pub = rospy.Publisher('/synced/camera5/usb_cam_5/image_raw', Image, queue_size=1)

    # ---- 시간 동기화 설정 ----
    ats = ApproximateTimeSynchronizer(
        [lidar_sub, camera1_sub, camera2_sub, camera3_sub, camera4_sub, camera5_sub],
        queue_size=10,
        slop=0.3
    )
    ats.registerCallback(sync_callback)

    rospy.spin()

if __name__ == '__main__':
    main()
