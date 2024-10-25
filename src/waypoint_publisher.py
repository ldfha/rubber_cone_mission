#!/usr/bin/env python3
import numpy as np
import rospy
import math
from geometry_msgs.msg import PoseStamped, Point
from rubber_cone_mission.msg import CentroidWithLabelArray
from visualization_msgs.msg import Marker, MarkerArray
from erp_driver.msg import erpCmdMsg, erpStatusMsg
from nav_msgs.msg import Path

class WaypointExtractor:
    def __init__(self):
        rospy.init_node('waypoint_extractor', anonymous=True)

        # 센트로이드 정보를 구독
        self.centroid_sub = rospy.Subscriber('/classified_centroids', CentroidWithLabelArray, self.centroid_callback)
        self.status_sub = rospy.Subscriber('/erp42_status', erpStatusMsg, self.status_callback)

        # 웨이포인트 퍼블리셔 (PoseStamped 및 MarkerArray용)
        self.path_pub = rospy.Publisher('/waypoint_path', Path, queue_size=10)
        self.waypoint_marker_pub = rospy.Publisher('/waypoint_markers', MarkerArray, queue_size=10)

        # 파라미터 설정
        self.centroid_threshold = 5.0  # 임계값 (거리가 너무 먼 센트로이드 무시)
        self.default_offset = 1.0  # 한쪽 센트로이드만 감지될 때 사용하는 오프셋
        self.steering_angle_deg = 0.0  # 조향각 (기본값 0)
        self.x_match_threshold = 0.3  # x 좌표 매칭 임계값

    def status_callback(self, msg):
        # 차량의 현재 속도 및 조향각을 업데이트
        self.current_steering_angle = (msg.steer / 2000.0) * 30  # ERP42의 조향각 범위(-2000 ~ 2000)에서 degree로 변환

    def centroid_callback(self, msg):
        # CentroidWithLabelArray에서 센트로이드들을 레이블에 따라 분류
        left_centroids = []
        right_centroids = []

        for centroid_with_label in msg.centroids:
            point = centroid_with_label.centroid
            label = centroid_with_label.label

            # 차량과 센트로이드 간의 거리 계산
            distance = math.sqrt(point.x**2 + point.y**2)
            # 임계값 필터링 (센트로이드가 너무 먼 경우 무시)
            if distance > self.centroid_threshold:
                continue

            if label == 'left':
                left_centroids.append(point)
            elif label == 'right':
                right_centroids.append(point)

        # 센트로이드들을 x 좌표 기준으로 정렬 (전방 기준)
        left_centroids.sort(key=lambda p: p.x)
        right_centroids.sort(key=lambda p: p.x)

        # 웨이포인트 계산 (양쪽 센트로이드가 존재할 때)
        if left_centroids and right_centroids:
            self.calculate_waypoints(left_centroids, right_centroids)
        # 왼쪽 센트로이드만 존재할 때
        elif left_centroids:
            rospy.logwarn("왼쪽 센트로이드만 감지되었습니다.")
            self.calculate_single_side_waypoint(left_centroids, "left")
        # 오른쪽 센트로이드만 존재할 때
        elif right_centroids:
            rospy.logwarn("오른쪽 센트로이드만 감지되었습니다.")
            self.calculate_single_side_waypoint(right_centroids, "right")
        else:
            rospy.logwarn("센트로이드가 감지되지 않았습니다.")

    def calculate_waypoints(self, left_centroids, right_centroids):
        waypoints = []

        # 센트로이드를 numpy 배열로 변환
        left_points = np.array([[p.x, p.y] for p in left_centroids])
        right_points = np.array([[p.x, p.y] for p in right_centroids])

        # LiDAR 기준 각 센트로이드까지의 거리 계산
        left_distances = np.linalg.norm(left_points, axis=1)
        right_distances = np.linalg.norm(right_points, axis=1)

        # 가장 가까운 센트로이드 인덱스 찾기
        left_closest_idx = np.argmin(left_distances)
        right_closest_idx = np.argmin(right_distances)

        # 가장 가까운 센트로이드들 추출
        left_closest_centroid = left_centroids[left_closest_idx]
        right_closest_centroid = right_centroids[right_closest_idx]

        # 거리가 더 먼 센트로이드 선택
        if left_distances[left_closest_idx] > right_distances[right_closest_idx]:
            reference_centroid = left_closest_centroid
            reference_side = 'left'
            opposite_centroids = right_centroids
            opposite_points = right_points
            left_used = [left_closest_idx]
            right_used = []
        else:
            reference_centroid = right_closest_centroid
            reference_side = 'right'
            opposite_centroids = left_centroids
            opposite_points = left_points
            left_used = []
            right_used = [right_closest_idx]

        # 반대쪽에서 x 좌표가 가장 가까운 센트로이드 찾기
        if len(opposite_centroids) > 0:
            x_diffs = np.abs(opposite_points[:, 0] - reference_centroid.x)
            matching_idx = np.argmin(x_diffs)
            matching_centroid = opposite_centroids[matching_idx]

            # 중점 계산하여 웨이포인트 추가
            mid_x = (reference_centroid.x + matching_centroid.x) / 2
            mid_y = (reference_centroid.y + matching_centroid.y) / 2
            waypoints.append(Point(mid_x, mid_y, 0.0))

            # 사용된 센트로이드 인덱스 저장
            if reference_side == 'left':
                left_used = [left_closest_idx]
                right_used = [matching_idx]
            else:
                left_used = [matching_idx]
                right_used = [right_closest_idx]
        else:
            # 반대쪽 센트로이드가 없는 경우 오프셋 적용
            if reference_side == 'left':
                waypoint_x = reference_centroid.x
                waypoint_y = reference_centroid.y - self.default_offset
            else:
                waypoint_x = reference_centroid.x
                waypoint_y = reference_centroid.y + self.default_offset
            waypoints.append(Point(waypoint_x, waypoint_y, 0.0))

        # 남은 센트로이드 처리
        left_indices = [i for i in range(len(left_centroids)) if i not in left_used]
        right_indices = [i for i in range(len(right_centroids)) if i not in right_used]

        # 남은 왼쪽 센트로이드 매칭
        for left_idx in left_indices:
            left_centroid = left_centroids[left_idx]
            if len(right_indices) > 0:
                x_diffs = np.abs(right_points[right_indices, 0] - left_centroid.x)
                min_x_diff = np.min(x_diffs)
                if min_x_diff < self.x_match_threshold:
                    matching_idx = right_indices[np.argmin(x_diffs)]
                    right_centroid = right_centroids[matching_idx]
                    # 중점 계산
                    mid_x = (left_centroid.x + right_centroid.x) / 2
                    mid_y = (left_centroid.y + right_centroid.y) / 2
                    waypoints.append(Point(mid_x, mid_y, 0.0))
                    right_indices.remove(matching_idx)
                else:
                    # 짝이 없는 경우 오프셋 적용
                    waypoint_x = left_centroid.x
                    waypoint_y = left_centroid.y - self.default_offset
                    waypoints.append(Point(waypoint_x, waypoint_y, 0.0))
            else:
                # 오른쪽 센트로이드가 없는 경우 오프셋 적용
                waypoint_x = left_centroid.x
                waypoint_y = left_centroid.y - self.default_offset
                waypoints.append(Point(waypoint_x, waypoint_y, 0.0))

        # 남은 오른쪽 센트로이드 매칭
        for right_idx in right_indices:
            right_centroid = right_centroids[right_idx]
            if len(left_indices) > 0:
                x_diffs = np.abs(left_points[left_indices, 0] - right_centroid.x)
                min_x_diff = np.min(x_diffs)
                if min_x_diff < self.x_match_threshold:
                    matching_idx = left_indices[np.argmin(x_diffs)]
                    left_centroid = left_centroids[matching_idx]
                    # 중점 계산
                    mid_x = (right_centroid.x + left_centroid.x) / 2
                    mid_y = (right_centroid.y + left_centroid.y) / 2
                    waypoints.append(Point(mid_x, mid_y, 0.0))
                    left_indices.remove(matching_idx)
                else:
                    # 짝이 없는 경우 오프셋 적용
                    waypoint_x = right_centroid.x
                    waypoint_y = right_centroid.y + self.default_offset
                    waypoints.append(Point(waypoint_x, waypoint_y, 0.0))
            else:
                # 왼쪽 센트로이드가 없는 경우 오프셋 적용
                waypoint_x = right_centroid.x
                waypoint_y = right_centroid.y + self.default_offset
                waypoints.append(Point(waypoint_x, waypoint_y, 0.0))

        # 웨이포인트 퍼블리시
        self.publish_waypoint_markers(waypoints)
        self.publish_waypoints(waypoints)

    def calculate_single_side_waypoint(self, centroids, side):
        waypoints = []
        for centroid in centroids:
            if side == "left":
                waypoint_x = centroid.x - self.default_offset / 2
                waypoint_y = centroid.y - self.default_offset
            elif side == "right":
                waypoint_x = centroid.x - self.default_offset / 2
                waypoint_y = centroid.y + self.default_offset 

            waypoints.append(Point(waypoint_x, waypoint_y, 0.0))

        self.publish_waypoint_markers(waypoints)
        self.publish_waypoints(waypoints)

    def publish_waypoints(self, waypoints):
        if waypoints:
            # 웨이포인트를 x 좌표 기준으로 정렬 (필요에 따라 변경)
            waypoints.sort(key=lambda p: p.x)

            path_msg = Path()
            path_msg.header.stamp = rospy.Time.now()
            path_msg.header.frame_id = "velodyne"  # 필요한 경우 적절한 프레임으로 변경

            for waypoint in waypoints:
                pose = PoseStamped()
                pose.header = path_msg.header
                pose.pose.position = waypoint
                pose.pose.orientation.w = 1.0  # 방향 정보가 없으면 기본값 사용
                path_msg.poses.append(pose)

            self.path_pub.publish(path_msg)
            rospy.loginfo(f"퍼블리시된 웨이포인트 수: {len(waypoints)}")

    def publish_waypoint_markers(self, waypoints):
        marker_array = MarkerArray()

        for idx, waypoint in enumerate(waypoints):
            marker = Marker()
            marker.header.frame_id = "velodyne"
            marker.header.stamp = rospy.Time.now()
            marker.ns = "waypoints"
            marker.id = idx
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD
            marker.pose.position = waypoint
            marker.pose.orientation.w = 1.0
            marker.scale.x = 0.2
            marker.scale.y = 0.2
            marker.scale.z = 0.2
            marker.color.r = 1.0
            marker.color.g = 1.0
            marker.color.b = 1.0
            marker.color.a = 1.0

            marker.lifetime = rospy.Duration(0.3)

            marker_array.markers.append(marker)

        self.waypoint_marker_pub.publish(marker_array)

if __name__ == '__main__':
    extractor = WaypointExtractor()
    rospy.spin()
