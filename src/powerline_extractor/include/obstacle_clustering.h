/*
这个类是用来进行数据预处理的：通过聚类找到并剔除传入的激光雷达中的点云的建筑部分
*/
#ifndef OBSTACLE_CLUSTERING_H
#define OBSTACLE_CLUSTERING_H

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/features/normal_3d.h>
#include <pcl/features/moment_of_inertia_estimation.h>
#include <pcl/filters/voxel_grid.h>
#include <ros/ros.h>
#include <vector>
#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>

struct BoundingBox {
    float x_min, y_min, z_min;
    float x_max, y_max, z_max;
};

class ObstacleClustering {
public:
    ObstacleClustering(ros::NodeHandle& nh);
    void process(const pcl::PointCloud<pcl::PointXYZI>::Ptr& input_cloud,
                 pcl::PointCloud<pcl::PointXYZI>::Ptr& output_cloud,
                 std::vector<BoundingBox>& excluded_regions);
private:
    ros::Publisher marker_pub_;
    double eps_;
    int min_samples_;
    double height_threshold_;
    double size_threshold_;
    double planarity_threshold_;
    int min_points_;

    void readParams(ros::NodeHandle& nh);
    void clusterPointCloud(const pcl::PointCloud<pcl::PointXYZI>::Ptr& cloud,
                           std::vector<pcl::PointIndices>& cluster_indices);
    void extractFeatures(const pcl::PointCloud<pcl::PointXYZI>::Ptr& cloud,
                         const std::vector<pcl::PointIndices>& cluster_indices,
                         std::vector<bool>& is_building,
                         std::vector<BoundingBox>& excluded_regions);
    void filterPointCloud(const pcl::PointCloud<pcl::PointXYZI>::Ptr& input_cloud,
                          const std::vector<bool>& is_building,
                          const std::vector<pcl::PointIndices>& cluster_indices,
                          pcl::PointCloud<pcl::PointXYZI>::Ptr& output_cloud);
    void publishBoundingBoxes(const std::vector<BoundingBox>& excluded_regions);
};

#endif // OBSTACLE_CLUSTERING_H

