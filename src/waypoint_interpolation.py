#!/usr/bin/env python3
import rospy
import numpy as np
from scipy.interpolate import CubicSpline
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped
from visualization_msgs.msg import Marker, MarkerArray

class WaypointInterpolator:
    def __init__(self):
        rospy.init_node('waypoint_interpolator', anonymous=True)

        # 웨이포인트를 구독
        self.waypoint_sub = rospy.Subscriber('/waypoint_path', Path, self.waypoint_callback)

        # 보간된 경로를 퍼블리시
        self.smoothed_path_pub = rospy.Publisher('/smoothed_path', Path, queue_size=10)

        # RViz에서 시각화를 위한 마커 퍼블리셔 추가
        self.path_marker_pub = rospy.Publisher('/path_markers', MarkerArray, queue_size=10)

    def waypoint_callback(self, msg):
        # 웨이포인트 좌표 추출
        x = []
        y = []
        for pose_stamped in msg.poses:
            x.append(pose_stamped.pose.position.x)
            y.append(pose_stamped.pose.position.y)

        # 웨이포인트 수에 따른 처리
        num_waypoints = len(x)
        if num_waypoints == 0:
            rospy.logwarn("웨이포인트가 없습니다.")
            return
        elif num_waypoints == 1:
            rospy.logwarn("웨이포인트가 하나뿐입니다. 해당 점만 퍼블리시합니다.")
            smoothed_path = self.create_path_message([x[0]], [y[0]])
            self.smoothed_path_pub.publish(smoothed_path)
            self.publish_path_markers(smoothed_path)
        elif num_waypoints == 2:
            rospy.loginfo("웨이포인트가 두 개입니다. 직선으로 연결합니다.")
            # 두 점 사이를 직선으로 연결
            num_points = 50  # 직선 상의 점 개수
            x_new = np.linspace(x[0], x[1], num=num_points)
            y_new = np.linspace(y[0], y[1], num=num_points)
            smoothed_path = self.create_path_message(x_new, y_new)
            self.smoothed_path_pub.publish(smoothed_path)
            self.publish_path_markers(smoothed_path)
        else:
            # 웨이포인트 수가 3개 이상인 경우 스플라인 보간 수행
            # x 좌표가 단조 증가하도록 정렬
            if not np.all(np.diff(x) > 0):
                sorted_indices = np.argsort(x)
                x = np.array(x)[sorted_indices]
                y = np.array(y)[sorted_indices]
            else:
                x = np.array(x)
                y = np.array(y)

            # 스플라인 보간 함수 생성
            spline = CubicSpline(x, y)

            # 보간된 경로 생성
            x_new = np.linspace(x[0], x[-1], num=100)  # 해상도에 따라 num 값 조정 가능
            y_new = spline(x_new)

            smoothed_path = self.create_path_message(x_new, y_new)
            self.smoothed_path_pub.publish(smoothed_path)
            self.publish_path_markers(smoothed_path)
            rospy.loginfo("스플라인 보간된 경로를 퍼블리시했습니다.")

    def create_path_message(self, x_points, y_points):
        path_msg = Path()
        path_msg.header.stamp = rospy.Time.now()
        path_msg.header.frame_id = "velodyne"  # 필요한 경우 적절한 프레임으로 변경

        for xi, yi in zip(x_points, y_points):
            pose = PoseStamped()
            pose.header = path_msg.header
            pose.pose.position.x = xi
            pose.pose.position.y = yi
            pose.pose.position.z = 0.0
            pose.pose.orientation.w = 1.0
            path_msg.poses.append(pose)

        return path_msg

    def publish_path_markers(self, path_msg):
        # RViz에서 시각화를 위한 마커 생성
        marker_array = MarkerArray()

        # Line Strip 마커 생성 (경로를 선으로 표시)
        line_marker = Marker()
        line_marker.header = path_msg.header
        line_marker.ns = "path_line"
        line_marker.id = 0
        line_marker.type = Marker.LINE_STRIP
        line_marker.action = Marker.ADD
        line_marker.scale.x = 0.1  # 선의 두께 설정
        line_marker.color.r = 1.0
        line_marker.color.g = 0.0
        line_marker.color.b = 0.0
        line_marker.color.a = 1.0

        # 경로의 점들을 마커에 추가
        for pose in path_msg.poses:
            point = pose.pose.position
            line_marker.points.append(point)

        marker_array.markers.append(line_marker)

        # 마커 퍼블리시
        self.path_marker_pub.publish(marker_array)

if __name__ == '__main__':
    try:
        interpolator = WaypointInterpolator()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
