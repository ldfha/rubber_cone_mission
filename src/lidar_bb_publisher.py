#!/usr/bin/env python3

import rospy
import random
import numpy as np
import sensor_msgs.point_cloud2 as pc2
from sensor_msgs.msg import PointCloud2, PointField
from sklearn.cluster import DBSCAN
from geometry_msgs.msg import Point
from rubber_cone_mission.msg import CentroidWithLabel, CentroidWithLabelArray  # Custom messages for centroids
from visualization_msgs.msg import Marker, MarkerArray

class LidarProcessor:
    def __init__(self):
        rospy.init_node('lidar_processor', anonymous=True)
        
        # Subscribers
        self.point_cloud_sub = rospy.Subscriber("/synced/velodyne_points", PointCloud2, self.point_cloud_callback, queue_size=1)
        
        # Publishers
        self.centroid_pub = rospy.Publisher("/centroid_info", CentroidWithLabelArray, queue_size=1)
        self.processed_points_pub = rospy.Publisher("/processed_points", PointCloud2, queue_size=1)
        self.clustered_points_pub = rospy.Publisher("/clustered_points", PointCloud2, queue_size=1)
        self.centroid_markers_pub = rospy.Publisher("/centroid_markers", MarkerArray, queue_size=1)
        self.text_markers_pub = rospy.Publisher("/centroid_text_markers", MarkerArray, queue_size=1)  # 텍스트 마커 퍼블리셔 추가
        
        # Parameters
        self.voxel_size = rospy.get_param("~voxel_size", 0.01)  # Voxel 그리드 크기
        self.crop_x = rospy.get_param("~crop_x", (0, 6))
        self.crop_y = rospy.get_param("~crop_y", (-3, 3))
        self.crop_z = rospy.get_param("~crop_z", (-0.5, 1))
        self.dbscan_eps = rospy.get_param("~dbscan_eps", 0.3)  # DBSCAN epsilon 파라미터
        self.dbscan_min_samples = rospy.get_param("~dbscan_min_samples", 20)  # DBSCAN min_samples 파라미터
        
        # 클러스터 크기 파라미터
        self.min_cluster_size = rospy.get_param("~min_cluster_size", 20)  # 최소 클러스터 크기
        self.max_cluster_size = rospy.get_param("~max_cluster_size", 1000)  # 최대 클러스터 크기

    def point_cloud_callback(self, msg):
        # Convert PointCloud2 to numpy array
        points = self.convert_point_cloud_to_numpy(msg)
        if points.shape[0] < 3:
            rospy.logwarn("Not enough points for processing.")
            return

        # Apply crop filter
        cropped_points = self.apply_crop_filter(points)
        if cropped_points.shape[0] < 3:
            rospy.logwarn("Not enough points after cropping.")
            return

        # Apply voxelization
        voxelized_points = self.apply_voxelization(cropped_points)
        if voxelized_points.shape[0] < 3:
            rospy.logwarn("Not enough points after voxelization.")
            return

        # Publish the processed point cloud
        processed_cloud_msg = pc2.create_cloud_xyz32(msg.header, voxelized_points)
        self.processed_points_pub.publish(processed_cloud_msg)

        # Perform clustering
        labels = self.perform_clustering(voxelized_points)

        # Publish clustered points
        self.publish_clustered_points(voxelized_points, labels, msg.header)

        # Compute and publish centroids with cluster size filtering
        self.publish_centroids(voxelized_points, labels, msg.header)

    def convert_point_cloud_to_numpy(self, cloud_msg):
        # Convert PointCloud2 to numpy array
        points_list = []
        for point in pc2.read_points(cloud_msg, field_names=("x", "y", "z"), skip_nans=True):
            points_list.append([point[0], point[1], point[2]])
        return np.array(points_list)

    def apply_crop_filter(self, points):
        x_min, x_max = self.crop_x
        y_min, y_max = self.crop_y
        z_min, z_max = self.crop_z

        mask = (
            (points[:, 0] >= x_min) & (points[:, 0] <= x_max) &
            (points[:, 1] >= y_min) & (points[:, 1] <= y_max) &
            (points[:, 2] >= z_min) & (points[:, 2] <= z_max)
        )
        return points[mask]

    def apply_voxelization(self, points):
        voxel_size = self.voxel_size
        voxel_indices = np.floor(points / voxel_size).astype(np.int32)
        coords, unique_indices = np.unique(voxel_indices, axis=0, return_index=True)
        return points[unique_indices]

    def perform_clustering(self, points):
        if points.shape[0] > 0:
            clustering = DBSCAN(eps=self.dbscan_eps, min_samples=self.dbscan_min_samples).fit(points)
            return clustering.labels_
        else:
            return np.array([])

    def publish_clustered_points(self, points, labels, header):
        unique_labels = set(labels)
        unique_labels.discard(-1)  # Remove noise label
        num_clusters = len(unique_labels)
        colors = self.get_colors(num_clusters)
        label_color_map = {label: color for label, color in zip(unique_labels, colors)}
        # For noise, set color to white
        label_color_map[-1] = (255, 255, 255)
        # Build point list with colors
        point_list = []
        for point, label in zip(points, labels):
            color = label_color_map.get(label, (255, 255, 255))
            r, g, b = color
            rgb = self.rgb_to_float(r, g, b)
            point_list.append([point[0], point[1], point[2], rgb])
        fields = [
            PointField('x', 0, PointField.FLOAT32, 1),
            PointField('y', 4, PointField.FLOAT32, 1),
            PointField('z', 8, PointField.FLOAT32, 1),
            PointField('rgb', 16, PointField.FLOAT32, 1),
        ]
        clustered_cloud_msg = pc2.create_cloud(header, fields, point_list)
        self.clustered_points_pub.publish(clustered_cloud_msg)

    def publish_centroids(self, points, labels, header):
        unique_labels = set(labels)
        unique_labels.discard(-1)  # Remove noise

        centroid_with_label_array = CentroidWithLabelArray()
        centroid_with_label_array.header = header

        marker_array = MarkerArray()
        text_marker_array = MarkerArray()
        marker_id = 0

        for cluster_id in unique_labels:
            cluster_points = points[labels == cluster_id]
            cluster_size = cluster_points.shape[0]

            # Filter clusters based on size
            if cluster_size < self.min_cluster_size or cluster_size > self.max_cluster_size:
                rospy.loginfo(f"Cluster {cluster_id} ignored due to size {cluster_size}.")
                continue

            # Compute centroid
            centroid = np.mean(cluster_points, axis=0)

            # Create CentroidWithLabel message
            centroid_msg = CentroidWithLabel()
            centroid_msg.centroid = Point(*centroid)
            centroid_msg.label = "unknown"  # Initialize as unknown; will be classified in another node
            centroid_with_label_array.centroids.append(centroid_msg)

            # Determine color based on cluster size
            max_size = self.max_cluster_size
            size_ratio = min(cluster_size / max_size, 1.0)  # Normalize between 0 and 1
            r = size_ratio
            g = 0.0
            b = 1.0 - size_ratio
            color = (r, g, b, 1.0)  # RGBA

            # Create Marker for visualization
            marker = Marker()
            marker.header = header
            marker.ns = "centroids"
            marker.id = marker_id
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD
            marker.pose.position = Point(*centroid)
            marker.pose.orientation.w = 1.0
            marker.scale.x = 0.3
            marker.scale.y = 0.3
            marker.scale.z = 0.3
            marker.color.r = color[0]
            marker.color.g = color[1]
            marker.color.b = color[2]
            marker.color.a = color[3]
            marker.lifetime = rospy.Duration(0)  # 무한한 lifetime 설정
            marker_array.markers.append(marker)

            marker_id += 1  # marker_id 증가

            # Create Text Marker for cluster ID and size
            text_marker = Marker()
            text_marker.header = header
            text_marker.ns = "centroid_info"
            text_marker.id = marker_id
            text_marker.type = Marker.TEXT_VIEW_FACING
            text_marker.action = Marker.ADD
            text_marker.pose.position = Point(*centroid)
            text_marker.pose.position.z += 0.5  # 클러스터 위에 텍스트 표시
            text_marker.pose.orientation.w = 1.0
            text_marker.scale.z = 0.4  # 텍스트 크기
            text_marker.color.r = 1.0
            text_marker.color.g = 1.0
            text_marker.color.b = 1.0
            text_marker.color.a = 1.0
            text_marker.text = f"ID: {cluster_id}\nSize: {cluster_size}"  # 클러스터 ID와 크기 표시
            text_marker.lifetime = rospy.Duration(0)  # 무한한 lifetime 설정
            text_marker_array.markers.append(text_marker)

            marker_id += 1  # marker_id 증가

        # Publish centroids
        if len(centroid_with_label_array.centroids) > 0:
            self.centroid_pub.publish(centroid_with_label_array)
            self.centroid_markers_pub.publish(marker_array)
            self.text_markers_pub.publish(text_marker_array)  # 텍스트 마커 퍼블리시
            rospy.loginfo(f"Published {len(centroid_with_label_array.centroids)} centroids.")
        else:
            rospy.loginfo("No centroids to publish after filtering.")

    def get_colors(self, num_colors):
        import random
        random.seed(0)  # For consistent colors
        colors = []
        for _ in range(num_colors):
            colors.append((random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)))
        return colors

    def rgb_to_float(self, r, g, b):
        import struct
        rgb = (int(r) << 16) | (int(g) << 8) | int(b)
        s = struct.pack('>I', rgb)
        rgb_float = struct.unpack('>f', s)[0]
        return rgb_float

if __name__ == "__main__":
    try:
        processor = LidarProcessor()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
