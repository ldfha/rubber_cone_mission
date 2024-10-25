#!/usr/bin/env python3
import rospy
import cv2
import numpy as np
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import matplotlib.pyplot as plt

# 전역 변수 설정
points = []  # 클릭한 좌표를 저장할 리스트
rect_drawn = False  # 사각형이 그려졌는지 확인
bridge = CvBridge()  # ROS 이미지 메시지를 OpenCV 형식으로 변환하기 위한 cv_bridge 객체
blurred_frame = None  # 블러링 된 이미지 저장 변수

# 마우스 클릭 이벤트 처리 함수
def click_event(event, x, y, flags, param):
    global points, rect_drawn, blurred_frame

    # 마우스 왼쪽 버튼을 클릭하면 좌표 저장
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(points) < 4:
            points.append((x, y))
            cv2.circle(blurred_frame, (x, y), 5, (0, 0, 255), -1)  # 클릭한 지점을 표시
            cv2.imshow("Blurred Image", blurred_frame)

        # 4개의 점을 클릭했을 때 사각형 그리기
        if len(points) == 4:
            rect_drawn = True
            # 사각형 그리기
            cv2.polylines(blurred_frame, [np.array(points, np.int32)], isClosed=True, color=(0, 255, 0), thickness=2)
            cv2.imshow("Blurred Image", blurred_frame)

            # 사각형 영역의 HSV 값 계산
            calculate_hsv_values(blurred_frame, points)

# 사각형 영역의 HSV 값 계산 함수
def calculate_hsv_values(image, points):
    global rect_drawn

    # 사각형 영역의 좌표 범위 계산 및 경계 설정
    x_min = max(min([p[0] for p in points]), 0)
    x_max = min(max([p[0] for p in points]), image.shape[1] - 1)
    y_min = max(min([p[1] for p in points]), 0)
    y_max = min(max([p[1] for p in points]), image.shape[0] - 1)

    # BGR에서 HSV로 변환
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # 사각형 영역의 HSV 값 추출
    hsv_roi = hsv_image[y_min:y_max, x_min:x_max]

    # HSV 값 계산
    mean_hsv = cv2.mean(hsv_roi)[:3]  # 평균값
    min_hsv = np.min(hsv_roi.reshape(-1, 3), axis=0)  # 최소값
    max_hsv = np.max(hsv_roi.reshape(-1, 3), axis=0)  # 최대값

    # 결과 출력
    print(f"HSV Mean: {mean_hsv}")
    print(f"HSV Min: {min_hsv}")
    print(f"HSV Max: {max_hsv}")

    # 추천 색상 범위 계산 및 출력
    suggested_hue_min = int(min_hsv[0])
    suggested_hue_max = int(max_hsv[0])
    print(f"Recommended Hue Range: {suggested_hue_min} ~ {suggested_hue_max}")

    # 선택한 영역을 새 창에 시각화
    visualize_selected_region(image, x_min, y_min, x_max, y_max)

    # HSV 히스토그램 시각화
    visualize_hsv_histogram(hsv_roi)

    # HSV 영역을 시각적으로 표시
    visualize_hsv_roi(hsv_roi)

    # 히스토그램 창을 닫을 때 초기화
    plt.close('all')
    reset_selection()  # 좌표와 플래그 초기화

# 선택한 영역을 시각화하는 함수
def visualize_selected_region(image, x_min, y_min, x_max, y_max):
    # 선택한 영역을 사각형으로 표시한 이미지 생성
    selected_region_image = image.copy()
    cv2.rectangle(selected_region_image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

    # 선택한 영역의 크기와 좌표 정보를 이미지에 표시
    label = f"Selected Area: x={x_min}, y={y_min}, w={x_max - x_min}, h={y_max - y_min}"
    cv2.putText(selected_region_image, label, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # 선택한 영역을 표시하는 창 생성
    cv2.imshow("Selected Region", selected_region_image)

# HSV 히스토그램 시각화 함수
def visualize_hsv_histogram(hsv_roi):
    # 각 HSV 채널의 히스토그램 계산
    h_values = hsv_roi[:, :, 0].flatten()
    s_values = hsv_roi[:, :, 1].flatten()
    v_values = hsv_roi[:, :, 2].flatten()

    # 히스토그램 시각화
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.hist(h_values, bins=180, range=(0, 180), color='red', alpha=0.7)
    plt.title('Hue Histogram')
    plt.xlabel('Hue Value')
    plt.ylabel('Frequency')

    plt.subplot(1, 3, 2)
    plt.hist(s_values, bins=256, range=(0, 256), color='green', alpha=0.7)
    plt.title('Saturation Histogram')
    plt.xlabel('Saturation Value')
    plt.ylabel('Frequency')

    plt.subplot(1, 3, 3)
    plt.hist(v_values, bins=256, range=(0, 256), color='blue', alpha=0.7)
    plt.title('Value Histogram')
    plt.xlabel('Value')
    plt.ylabel('Frequency')

    plt.show()  # 히스토그램 창 열기

# HSV 영역 시각화 함수
def visualize_hsv_roi(hsv_roi):
    # HSV 영역을 BGR로 변환하여 시각화
    bgr_roi = cv2.cvtColor(hsv_roi, cv2.COLOR_HSV2BGR)
    cv2.imshow("Selected HSV Region", bgr_roi)

# 선택 초기화 함수
def reset_selection():
    global points, rect_drawn, blurred_frame

    points = []  # 선택한 좌표 초기화
    rect_drawn = False  # 사각형 그린 상태 초기화
    print("Selection reset. You can now select a new region.")

# ROS 이미지 콜백 함수
def image_callback(ros_image):
    global blurred_frame

    # ROS Image 메시지를 OpenCV 이미지로 변환
    frame = bridge.imgmsg_to_cv2(ros_image, "bgr8")

    # Gaussian Blur 적용 (커널 크기: 11x11, 표준 편차: 5)
    blurred_frame = cv2.GaussianBlur(frame, (11, 11), 5)

    # 블러된 이미지와 원본 이미지를 같이 표시
    cv2.imshow("Original Image", frame)
    cv2.imshow("Blurred Image", blurred_frame)

    # 마우스 클릭 이벤트 연결
    cv2.setMouseCallback("Blurred Image", click_event)

    # OpenCV 창에서 'q' 키 입력시 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        rospy.signal_shutdown("User Exit")

# 메인 함수
def main():
    # ROS 노드 초기화
    rospy.init_node('hsv_roi_extractor', anonymous=True)

    # 이미지 토픽 구독자 설정 (사용자의 이미지 토픽으로 변경)
    rospy.Subscriber("/camera1/usb_cam_1/image_raw", Image, image_callback)

    # ROS 루프 실행
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")

    # 모든 OpenCV 창 닫기
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
