cmake_minimum_required(VERSION 3.0.2)
project(rubber_cone_mission)

# Find catkin macros and libraries
find_package(catkin REQUIRED COMPONENTS
  geometry_msgs
  pcl_ros
  rospy
  sensor_msgs
  std_msgs
  tf
  vision_msgs
  erp_driver        # ERP42 차량 제어 관련 메시지 추가
  cv_bridge         # YOLOv9에서 카메라 이미지를 처리하기 위해 필요
  visualization_msgs # RViz에서 시각화를 위해 필요
  message_generation # 메시지 생성 관련 의존성 추가
)

###################################
## 메시지 파일 경로 지정 및 메시지 생성 ##
###################################
# 메시지 파일 경로 지정
add_message_files(
  FILES
  BoundingBox.msg
  BoundingBoxArray.msg
  CentroidWithLabel.msg
  CentroidWithLabelArray.msg
  ProjectedBoundingBox.msg
  ProjectedBoundingBoxArray.msg
)

## 의존성 추가 (geometry_msgs, std_msgs)
generate_messages(
  DEPENDENCIES
  std_msgs
  geometry_msgs
  vision_msgs
  sensor_msgs
)

###################################
## catkin specific configuration ##
###################################
catkin_package(
  CATKIN_DEPENDS message_runtime geometry_msgs pcl_ros rospy sensor_msgs std_msgs tf vision_msgs erp_driver cv_bridge visualization_msgs
)

###########
## Build ##
###########
include_directories(
  ${catkin_INCLUDE_DIRS}
)

#############
## Install ##
#############
## Python 스크립트 설치
catkin_install_python(PROGRAMS
  src/waypoint_publisher.py
  src/command_publisher.py
  src/centroid_pup_color.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

#############
## Testing ##
#############
## (Optional) 테스트 관련 설정
