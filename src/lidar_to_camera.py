#!/usr/bin/env python3

import rospy
import cv2
import numpy as np
from sensor_msgs.msg import Image, PointCloud2
from sensor_msgs import point_cloud2
from geometry_msgs.msg import Point
from visualization_msgs.msg import Marker, MarkerArray
from rubber_cone_mission.msg import ProjectedBoundingBoxArray
from cv_bridge import CvBridge

class CameraOverlayVisualizer:
    def __init__(self):
        rospy.init_node('camera_overlay_visualizer', anonymous=True)

        # CvBridge 초기화
        self.bridge = CvBridge()

        # 카메라 이미지 구독자 설정
        self.camera3_sub = rospy.Subscriber('/camera4/usb_cam_4/image_raw', Image, self.camera3_callback)
        self.camera4_sub = rospy.Subscriber('/camera5/usb_cam_5/image_raw', Image, self.camera4_callback)

        # 포인트 클라우드 데이터 구독자
        self.pointcloud_sub = rospy.Subscriber('/processed_points', PointCloud2, self.pointcloud_callback)

        # 투영된 바운딩 박스 구독자
        self.bbox_sub = rospy.Subscriber('/projected_bounding_boxes', ProjectedBoundingBoxArray, self.bbox_callback)

        # 카메라별 이미지 퍼블리셔
        self.image_pub_camera4 = rospy.Publisher('/camera4_image_overlay', Image, queue_size=10)
        self.image_pub_camera5 = rospy.Publisher('/camera5_image_overlay', Image, queue_size=10)

        # RViz에서 마커를 그리기 위한 퍼블리셔
        self.marker_pub = rospy.Publisher('/bounding_box_markers', MarkerArray, queue_size=10)

        # 최신 카메라 이미지 저장
        self.camera_images = {'camera4': None, 'camera5': None}

        # 최신 바운딩 박스 저장
        self.projected_bboxes = []

        # 포인트 클라우드 데이터 저장
        self.point_cloud_data = None

        # 카메라 캘리브레이션 파라미터 설정
        self.camera_params = {
            'camera4': {
                'camera_matrix': np.array([[528.9061800582897, 0, 291.7429233316773],
                                           [0, 522.8518353122911, 251.8800805745865],
                                           [0, 0, 1]]),
                'R': np.array([[-0.374119021599725, -0.927311067925970, 0.011364021262813],
               [-0.083948359573392, 0.021659879670429, -0.996234672422918],
               [0.923573294656250, -0.373664331873755, -0.085949615956904]]),
                'T': np.array([-0.432476342505947, 0.197143876577278, 0.448439624086928])
            },
            'camera5': {
                'camera_matrix': np.array([[573.2739492076811, 0, 290.1657508033189],
                                           [0, 574.9742361244955, 225.5800631193866],
                                           [0, 0, 1]]),
                'R': np.array([[0.460996596642375, -0.887374140579444, 0.007019438370723],
                               [-0.051360339356610, -0.034577199516021, -0.998081425944198],
                               [0.885914360096574, 0.459751619795431, -0.061515808294750]]),
                'T': np.array([0.498953116398469, 0.363073685499183, 0.784810307371323])
            }
        }

    def camera3_callback(self, msg):
        # camera3 이미지 저장
        self.camera_images['camera4'] = msg

    def camera4_callback(self, msg):
        # camera4 이미지 저장
        self.camera_images['camera5'] = msg

    def pointcloud_callback(self, pointcloud_msg):
        # 포인트 클라우드 데이터 저장
        self.point_cloud_data = pointcloud_msg

    def bbox_callback(self, bbox_array):
        # 바운딩 박스 저장
        self.projected_bboxes = bbox_array.boxes

        # 카메라 이미지 위에 바운딩 박스 덮어쓰기
        self.overlay_bounding_boxes()

    def overlay_bounding_boxes(self):
        # 각각의 카메라에 대해 이미지를 처리
        for camera_id, image_msg in self.camera_images.items():
            if image_msg is None:
                continue  # 이미지가 없으면 스킵

            # Image 메시지를 OpenCV 이미지로 변환
            cv_image = self.bridge.imgmsg_to_cv2(image_msg, desired_encoding='bgr8')

            # 포인트 클라우드 데이터를 카메라 이미지 위에 투영
            if self.point_cloud_data:
                camera_params = self.camera_params[camera_id]
                cv_image = self.project_pointcloud_onto_image(cv_image, self.point_cloud_data,
                                                              camera_params['camera_matrix'],
                                                              camera_params['R'],
                                                              camera_params['T'])

            # 바운딩 박스 및 포인트 클라우드 투영 결과 오버레이
            self.add_bounding_boxes_to_image(cv_image, camera_id)

            # OpenCV 이미지를 다시 ROS Image 메시지로 변환
            overlayed_image_msg = self.bridge.cv2_to_imgmsg(cv_image, encoding='bgr8')
            overlayed_image_msg.header.frame_id = camera_id

            # 카메라별로 이미지 퍼블리시
            if camera_id == 'camera4':
                self.image_pub_camera4.publish(overlayed_image_msg)
            elif camera_id == 'camera5':
                self.image_pub_camera5.publish(overlayed_image_msg)

            # RViz 마커를 사용하여 바운딩 박스 표시
            self.publish_bounding_box_markers(camera_id)

    def project_pointcloud_onto_image(self, image, point_cloud, camera_matrix, R, T):
        """포인트 클라우드 데이터를 카메라 이미지 위에 투영"""
        projected_image = image.copy()
        
        # 포인트 클라우드 데이터에서 포인트 추출
        for point in point_cloud2.read_points(point_cloud, skip_nans=True):
            # LiDAR 좌표계의 점을 가져옴
            point_lidar = np.array([point[0], point[1], point[2]])

            # 카메라 좌표계로 변환
            point_camera = R @ point_lidar + T

            # 카메라 앞에 있는 점만 처리
            if point_camera[2] <= 0:
                continue

            # 2D 이미지 좌표로 변환
            point_image = camera_matrix @ point_camera
            u = int(point_image[0] / point_image[2])
            v = int(point_image[1] / point_image[2])

            # 이미지 범위 내의 좌표만 표시
            if 0 <= u < projected_image.shape[1] and 0 <= v < projected_image.shape[0]:
                cv2.circle(projected_image, (u, v), 1, (0, 255, 0), -1)  # 초록색 점으로 표시

        return projected_image

    def add_bounding_boxes_to_image(self, image, camera_id):
        """바운딩 박스를 이미지에 오버레이"""
        for bbox in self.projected_bboxes:
            if bbox.camera_id != camera_id:
                continue  # 카메라 ID가 일치하지 않으면 스킵

            # 바운딩 박스 좌표 추출
            x_min = bbox.x_min
            y_min = bbox.y_min
            x_max = bbox.x_max
            y_max = bbox.y_max

            # 사각형 그리기
            if bbox.label == 'left':
                color = (0, 255, 255)  # 노란색 (left)
            else:
                color = (255, 0, 0)  # 파란색 (right)

            # 바운딩 박스를 이미지에 그리기
            cv2.rectangle(image, (x_min, y_min), (x_max, y_max), color, 2)
            cv2.putText(image, bbox.label, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    def publish_bounding_box_markers(self, camera_id):
        """RViz에서 바운딩 박스를 마커로 시각화"""
        marker_array = MarkerArray()
        marker_id = 0

        for bbox in self.projected_bboxes:
            if bbox.camera_id != camera_id:
                continue  # 카메라 ID가 일치하지 않으면 스킵

            marker = Marker()
            marker.header.frame_id = camera_id
            marker.header.stamp = rospy.Time.now()
            marker.ns = camera_id
            marker.id = marker_id
            marker.type = Marker.LINE_STRIP
            marker.action = Marker.ADD
            marker.scale.x = 0.02  # 선의 두께 설정

            p1 = self.create_point(bbox.x_min, bbox.y_min)
            p2 = self.create_point(bbox.x_max, bbox.y_min)
            p3 = self.create_point(bbox.x_max, bbox.y_max)
            p4 = self.create_point(bbox.x_min, bbox.y_max)

            marker.points = [p1, p2, p3, p4, p1]

            if bbox.label == 'left':
                marker.color.r = 1.0  # 노란색 (left)
                marker.color.g = 1.0
                marker.color.b = 0.0
            else:
                marker.color.r = 0.0  # 파란색 (right)
                marker.color.g = 0.0
                marker.color.b = 1.0
            marker.color.a = 1.0

            marker_array.markers.append(marker)
            marker_id += 1

        self.marker_pub.publish(marker_array)

    def create_point(self, x, y):
        """2D 이미지 좌표를 사용하여 포인트 생성"""
        point = Point()
        point.x = float(x) / 100.0  # RViz의 스케일에 맞게 좌표 조정
        point.y = float(y) / 100.0
        point.z = 0.0
        return point

if __name__ == '__main__':
    try:
        visualizer = CameraOverlayVisualizer()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
