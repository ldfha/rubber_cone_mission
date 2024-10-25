#!/usr/bin/env python3

import rospy
import numpy as np
from sensor_msgs.msg import Image
from geometry_msgs.msg import Point
from rubber_cone_mission.msg import CentroidWithLabel, CentroidWithLabelArray  # Custom messages for centroids
from visualization_msgs.msg import Marker, MarkerArray
from cv_bridge import CvBridge, CvBridgeError

import cv2

class CentroidColorClassifier:
    def __init__(self):
        rospy.init_node('centroid_color_classifier', anonymous=True)
        
        # Parameters
        self.kernel_size = 51  # 픽셀 영역 크기 (홀수로 설정)
        self.threshold = 0.4   # 색상 분류 임계값
        
        # Camera parameters dictionary
        self.camera_params = {
            'left': {  # 'left'에 해당하는 camera2 (2번 카메라)
                'camera_matrix': np.array([[5.201364791416012e+02, 0, 3.216949161493492e+02],
                                           [0, 5.318044062657257e+02, 1.977774069204301e+02],
                                           [0, 0, 1]]),
                'R': np.array([[0.394230287631315, -0.918950222405988, -0.010628690141133],
                               [-0.455883885053441, -0.185506282332136, -0.870492563187010],
                               [0.797967645749320, 0.348019982120704, -0.492066792602379]]),
                'T': np.array([0.385472551707401, 0.574845930983101, 1.266383264744131])
            },
            'right': {  # 'right'에 해당하는 camera1 (1번 카메라)
                'camera_matrix': np.array([[4.327558922371229e+02, 0, 3.248154270949172e+02],
                                           [0, 4.393889014615702e+02, 2.392671823824636e+02],
                                           [0, 0, 1]]),
                'R': np.array([[-0.354126260156503, -0.935152839833323, 0.009151940736245],
                               [-0.519345917018068, 0.188510112079435, -0.833512900992217],
                               [0.777736723026649, -0.299921829474731, -0.552424190147661]]),
                'T': np.array([-0.396962199403921, 0.604084261640270, 0.751677189083849])
            }
        }
        
        # CvBridge
        self.bridge = CvBridge()
        
        # Subscribers
        self.centroid_sub = rospy.Subscriber("/centroid_info", CentroidWithLabelArray, self.centroid_callback, queue_size=1)
        self.image_sub_left = rospy.Subscriber('/synced/camera2/usb_cam_2/image_raw', Image, self.image_callback_left)
        self.image_sub_right = rospy.Subscriber('/synced/camera1/usb_cam_1/image_raw', Image, self.image_callback_right)
        
        # Publishers
        self.classified_centroid_pub = rospy.Publisher('/classified_centroids', CentroidWithLabelArray, queue_size=1)
        self.centroid_marker_pub = rospy.Publisher('/classified_centroid_markers', MarkerArray, queue_size=1)
        self.annotated_image_pub_left = rospy.Publisher('/camera2/image_annotated', Image, queue_size=1)
        self.annotated_image_pub_right = rospy.Publisher('/camera1/image_annotated', Image, queue_size=1)
        
        # Data storage
        self.lidar_centroids = None
        self.current_image_left = None
        self.current_image_right = None

        
    def centroid_callback(self, centroid_array):
        if centroid_array is None or len(centroid_array.centroids) == 0:
            rospy.logwarn("Received empty centroid array.")
            return
        self.lidar_centroids = centroid_array
        self.process_centroids()
        
    def image_callback_left(self, img_msg):
        try:
            self.current_image_left = self.bridge.imgmsg_to_cv2(img_msg, "bgr8")
        except CvBridgeError as e:
            rospy.logerr(f"CvBridge Error (left camera): {e}")
            self.current_image_left = None

    def image_callback_right(self, img_msg):
        try:
            self.current_image_right = self.bridge.imgmsg_to_cv2(img_msg, "bgr8")
        except CvBridgeError as e:
            rospy.logerr(f"CvBridge Error (right camera): {e}")
            self.current_image_right = None
        
    def process_centroids(self):
        if self.lidar_centroids is None:
            rospy.loginfo("No centroid data to process.")
            return  # 필요한 데이터가 모두 있어야 진행

        # 이미지가 둘 다 준비되어 있는지 확인
        if self.current_image_left is None or self.current_image_right is None:
            rospy.loginfo("Waiting for images from both cameras.")
            return
        
        classified_centroid_array = CentroidWithLabelArray()
        classified_centroid_array.header.stamp = rospy.Time.now()
        classified_centroid_array.header.frame_id = "map"
        
        # 마커를 위한 설정
        marker_array = MarkerArray()
        marker_id = 0
        
        # 복사본 이미지 생성 (원본을 유지하기 위해)
        annotated_image_left = self.current_image_left.copy()
        annotated_image_right = self.current_image_right.copy()
        
        for centroid_msg in self.lidar_centroids.centroids:
            centroid_point = np.array([centroid_msg.centroid.x, centroid_msg.centroid.y, centroid_msg.centroid.z])
            
            labels_detected = []  # 감지된 라벨을 저장할 리스트
            
            # 왼쪽 카메라 (camera5) 투영 및 색상 분류
            uv_left = self.project_point_to_image(centroid_point, 'left')
            if uv_left is not None:
                u_left, v_left = uv_left
                label_left = self.classify_cone_color(annotated_image_left, u_left, v_left)
                if label_left in ['left', 'right']:
                    labels_detected.append(label_left)
                    # 시각화: 원과 사각형 그리기
                    color = (0, 255, 255) if label_left == 'left' else (255, 0, 0)  # 노란색 또는 파란색
                    cv2.circle(annotated_image_left, (u_left, v_left+10), 5, color, 2)
                    cv2.rectangle(annotated_image_left, 
                                  (u_left - self.kernel_size//2, v_left - self.kernel_size//2),
                                  (u_left + self.kernel_size//2, v_left + self.kernel_size//2),
                                  color, 2)
            
            # 오른쪽 카메라 (camera4) 투영 및 색상 분류
            uv_right = self.project_point_to_image(centroid_point, 'right')
            if uv_right is not None:
                u_right, v_right = uv_right
                label_right = self.classify_cone_color(annotated_image_right, u_right, v_right)
                if label_right in ['left', 'right']:
                    labels_detected.append(label_right)
                    # 시각화: 원과 사각형 그리기
                    color = (0, 255, 255) if label_right == 'left' else (255, 0, 0)  # 노란색 또는 파란색
                    cv2.circle(annotated_image_right, (u_right, v_right+10), 5, color, 2)
                    cv2.rectangle(annotated_image_right, 
                                  (u_right - self.kernel_size//2, v_right - self.kernel_size//2),
                                  (u_right + self.kernel_size//2, v_right + self.kernel_size//2),
                                  color, 2)
            
            # 라벨 결정: 'left'가 감지되면 'left', 아니면 'right'가 감지되면 'right'
            # 둘 다 감지되면 'left' 우선
            label = 'unknown'
            if 'left' in labels_detected:
                label = 'left'
            elif 'right' in labels_detected:
                label = 'right'
            
            # 분류된 라벨에 따라 센트로이드 퍼블리시
            if label in ['left', 'right']:
                # CentroidWithLabel 메시지 생성
                classified_centroid = CentroidWithLabel()
                classified_centroid.centroid = centroid_msg.centroid
                classified_centroid.label = label
                classified_centroid_array.centroids.append(classified_centroid)
                
                # 마커 생성
                marker = Marker()
                marker.header.frame_id = "velodyne"
                marker.id = marker_id
                marker.type = Marker.SPHERE
                marker.action = Marker.ADD
                marker.pose.position = centroid_msg.centroid
                marker.pose.orientation.w = 1.0
                marker.scale.x = 0.2
                marker.scale.y = 0.2
                marker.scale.z = 0.2
                if label == 'left':
                    marker.color.r = 1.0  # 노란색
                    marker.color.g = 1.0
                    marker.color.b = 0.0
                elif label == 'right':
                    marker.color.r = 0.0  # 파란색
                    marker.color.g = 0.0
                    marker.color.b = 1.0
                marker.color.a = 1.0
                marker.lifetime = rospy.Duration(0.3)
                marker_array.markers.append(marker)
                marker_id += 1
            else:
                rospy.loginfo("Centroid ignored due to color mismatch or projection failure.")
        
        # 퍼블리시
        if len(classified_centroid_array.centroids) > 0:
            self.classified_centroid_pub.publish(classified_centroid_array)
            self.centroid_marker_pub.publish(marker_array)
            rospy.loginfo(f"Number of published centroids: {len(classified_centroid_array.centroids)}")
        else:
            rospy.loginfo("No centroids classified.")
        
        # 시각화된 이미지 퍼블리시
        try:
            annotated_msg_left = self.bridge.cv2_to_imgmsg(annotated_image_left, "bgr8")
            annotated_msg_right = self.bridge.cv2_to_imgmsg(annotated_image_right, "bgr8")
            self.annotated_image_pub_left.publish(annotated_msg_left)
            self.annotated_image_pub_right.publish(annotated_msg_right)
        except CvBridgeError as e:
            rospy.logerr(f"CvBridge Error during image publishing: {e}")
        
        # 데이터 초기화
        self.lidar_centroids = None
        
    def project_point_to_image(self, point_lidar, camera_side):
        # 카메라 파라미터 가져오기
        camera_matrix = self.camera_params[camera_side]['camera_matrix']
        R = self.camera_params[camera_side]['R']
        T = self.camera_params[camera_side]['T']
        
        # LiDAR 좌표를 카메라 좌표로 변환
        point_camera = R @ point_lidar + T

        # 카메라 앞에 있는 점만 처리
        if point_camera[2] > 0:
            # 2D 이미지 평면으로 투영
            uv = camera_matrix @ point_camera
            u = int(uv[0] / uv[2])
            v = int(uv[1] / uv[2])
            
            # 이미지 경계 체크
            if camera_side == 'left':
                image = self.current_image_left
            else:
                image = self.current_image_right
                
            height, width, _ = image.shape
            if 0 <= u < width and 0 <= v < height:
                return (u, v)
            else:
                return None
        else:
            return None
        
    def adjust_gamma(self, image, gamma=1.0):
        invGamma = 1.0 / gamma
        table = np.array([
            ((i / 255.0) ** invGamma) * 255
            for i in np.arange(256)]).astype("uint8")
        return cv2.LUT(image, table)


    
    def classify_cone_color(self, image, u, v):
        # 주변 픽셀 영역 설정
        half_kernel = self.kernel_size // 2
        u_start = max(u - half_kernel, 0)
        u_end = min(u + half_kernel, image.shape[1] - 1)
        v_start = max(v, 0)
        v_end = min(v + self.kernel_size+15, image.shape[0] - 1)
        
        # 주변 픽셀 색상 추출
        pixel_region = image[v_start:v_end+1, u_start:u_end+1]
        
        if pixel_region.size == 0:
            return 'unknown'
        
        # **감마 보정 적용**
        pixel_region = self.adjust_gamma(pixel_region, gamma=1.2)
        
        # **Gaussian Blur 적용**
        pixel_region_blur = cv2.GaussianBlur(pixel_region, (5, 5), 0)
        
        # **HSV 색상 공간으로 변환**
        hsv_region = cv2.cvtColor(pixel_region_blur, cv2.COLOR_BGR2HSV)
        
        # **파란색과 노란색 범위 설정**
        yellow_lower = np.array([20, 20, 50])
        yellow_upper = np.array([50, 255, 255])
        blue_lower = np.array([80, 20, 30])  # 채도와 명도 하한값을 낮춤
        blue_upper = np.array([170, 255, 255])
        
        # **마스크 생성**
        yellow_mask = cv2.inRange(hsv_region, yellow_lower, yellow_upper)
        blue_mask = cv2.inRange(hsv_region, blue_lower, blue_upper)
        
        # **색상 비율 계산**
        total_pixels = hsv_region.shape[0] * hsv_region.shape[1]
        yellow_ratio = np.sum(yellow_mask > 0) / total_pixels
        blue_ratio = np.sum(blue_mask > 0) / total_pixels
        
        # **임계값 설정 및 분류**
        if yellow_ratio > self.threshold:
            return 'left'  # 노란색
        elif blue_ratio > self.threshold:
            return 'right'  # 파란색
        else:
            return 'unknown'



if __name__ == '__main__':
    try:
        classifier = CentroidColorClassifier()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
