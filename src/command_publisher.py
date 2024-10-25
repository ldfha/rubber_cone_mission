#!/usr/bin/env python3
import rospy
import math
from collections import deque
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path
from erp_driver.msg import erpCmdMsg, erpStatusMsg

class CommandPublisher:
    def __init__(self):
        rospy.init_node('command_publisher', anonymous=True)
        self.ctrl_cmd_msg = erpCmdMsg()

        # Path 메시지를 구독하도록 수정
        self.path_sub = rospy.Subscriber('/waypoint_path', Path, self.path_callback)
        self.status_sub = rospy.Subscriber('/erp42_status', erpStatusMsg, self.status_callback)

        self.erp_cmd_pub = rospy.Publisher('/erp42_ctrl_cmd', erpCmdMsg, queue_size=1)

        # 차량의 현재 상태 초기화
        self.current_position = [0.0, 0.0]  # 차량의 위치를 (0, 0)으로 고정
        self.current_yaw = 0.0  # 차량의 진행 방향을 0도로 고정 (X축 방향)
        self.current_speed = 0  # 현재 차량 속도
        self.current_steering_angle = 0  # 현재 조향각 (degree 단위)
        self.previous_steering_angle = 0  # 이전 주기에서의 조향각

        self.lat_err = deque(maxlen=100)
        self.PID_steer = self.PID()

        self.waypoints = []  # 웨이포인트 리스트
        self.waypoint_index = 0  # 현재 추종 중인 웨이포인트 인덱스

    def status_callback(self, msg):
        # 차량의 현재 속도 및 조향각을 업데이트
        self.current_speed = msg.speed  # 현재 차량의 속도 (0~255 범위)
        self.current_steering_angle = (msg.steer / 2000.0) * 30  # ERP42의 조향각 범위(-2000 ~ 2000)에서 degree로 변환

    def path_callback(self, msg):
        # Path 메시지에서 웨이포인트 리스트 추출
        self.waypoints = []
        for pose_stamped in msg.poses:
            x = pose_stamped.pose.position.x
            y = pose_stamped.pose.position.y
            self.waypoints.append([x, y])

        # 웨이포인트 인덱스 초기화
        self.waypoint_index = 0

        # 웨이포인트 추종 시작
        self.follow_waypoints()

    def follow_waypoints(self):
        rate = rospy.Rate(10)  # 제어 주기 설정 (10Hz)
        while not rospy.is_shutdown() and self.waypoint_index < len(self.waypoints):
            waypoint_x, waypoint_y = self.waypoints[self.waypoint_index]

            # 차량의 위치는 항상 (0, 0)이므로 웨이포인트까지의 상대 위치는 웨이포인트의 좌표 그대로 사용
            dx = waypoint_x - self.current_position[0]  # dx = waypoint_x - 0 = waypoint_x
            dy = waypoint_y - self.current_position[1]  # dy = waypoint_y - 0 = waypoint_y
            dist = math.sqrt(dx**2 + dy**2)

            # 웨이포인트에 도달하면 다음 웨이포인트로 이동
            if dist < 0.5:  # 도달 거리 임계값 (예: 0.5m)
                self.waypoint_index += 1
                continue

            # Waypoint까지의 각도 계산 (라디안 값)
            angle_to_waypoint = math.atan2(dy, dx)  # 차량의 진행 방향은 0도(X축)이므로, 현재 yaw는 고려하지 않음
            angle_error = angle_to_waypoint  # 현재 yaw가 0이므로 각도 에러는 angle_to_waypoint 그대로

            # 각도 에러를 -pi ~ pi 범위로 정규화
            angle_error = (angle_error + math.pi) % (2 * math.pi) - math.pi

            # 조향각 계산(deg)
            wheel_base = 1.04  # 차량의 휠베이스 길이 (실제 차량에 맞게 수정)
            delta = math.degrees(math.atan2(2 * wheel_base * math.sin(angle_error), dist))

            # 최종 조향각 계산 (ERP42의 조향각 범위에 맞춰 변환, -30도 ~ 30도 사이로 제한)
            steering_angle = delta
            steering_angle_deg = max(min(steering_angle, 30), -30)  # -30도 ~ 30도 사이로 제한

            # PID 제어
            delta_err = steering_angle_deg - self.current_steering_angle  # 에러 정의
            steer_pid = self.PID_steer.control(delta_err)
            steer_pid = max(min(steer_pid, 30), -30)

            # ERP42의 조향각 범위로 변환 (-1800 ~ 1800)
            steer_cmd = steer_pid
            steering_angle_cmd = int((steer_cmd / 30.0) * 1800)

            # 곡률 계산
            k = abs(2 * math.sin(angle_error) / dist)

            # 곡률에 따라 속도를 가변적으로 설정
            if k < 0.1:
                speed = 80
            elif k < 0.2:
                speed = 45
            else:
                speed = 30

            self.lat_err.append(dy)
            avg_lat_err = sum(self.lat_err) / len(self.lat_err)

            # ERP42 제어 명령 메시지 생성
            self.ctrl_cmd_msg.steer = -steering_angle_cmd  # 조향 명령 설정
            self.ctrl_cmd_msg.speed = speed  # 각도에 따라 동적으로 설정된 속도 값 사용
            self.ctrl_cmd_msg.gear = 0  # 기어 설정 (0: 전진)
            self.ctrl_cmd_msg.e_stop = False  # 비상 정지 설정 (False: 비상 정지 해제)
            self.ctrl_cmd_msg.brake = 0  # 브레이크 해제

            # 제어 명령 퍼블리시
            self.erp_cmd_pub.publish(self.ctrl_cmd_msg)
            rospy.loginfo(f"Published command: speed={speed}, steer={-steering_angle_cmd}, curvature={k}")
            rospy.loginfo(f"Average Lateral offset = {avg_lat_err}")

            # 이전 조향각 업데이트
            self.previous_steering_angle = self.current_steering_angle

            rate.sleep()

    class PID:
        def __init__(self):
            self.kp = 1.0
            self.ki = 0.001
            self.kd = 0.001
            self.Pterm = 0.0
            self.Iterm = 0.0
            self.Dterm = 0.0
            self.prev_error = 0.0
            self.dt = 0.1

        def control(self, error):
            self.Pterm = self.kp * error
            self.Iterm += error * self.dt
            self.Dterm = self.kd * (error - self.prev_error) / self.dt

            self.prev_error = error
            output = self.Pterm + self.ki * self.Iterm + self.Dterm
            rospy.loginfo(f"P_term: {self.Pterm}, I_term = {self.Iterm}, D_Term = {self.Dterm}")
            return output

if __name__ == '__main__':
    try:
        command_publisher = CommandPublisher()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
