#include "obstacle_clustering.h"
#include <pcl/filters/extract_indices.h>
#include <pcl/common/common.h>
#include <pcl/common/centroid.h>
#include <Eigen/Dense>
#include <omp.h> // 添加 OpenMP 头文件
#include <Eigen/Dense> 

ObstacleClustering::ObstacleClustering(ros::NodeHandle& nh) {
    marker_pub_ = nh.advertise<visualization_msgs::MarkerArray>("building_boxes", 1);
    readParams(nh);
}

void ObstacleClustering::readParams(ros::NodeHandle& nh) {
    nh.param("obstacle_clustering/eps", eps_, 0.3);
    nh.param("obstacle_clustering/min_samples", min_samples_, 20);
    nh.param("obstacle_clustering/height_threshold", height_threshold_, 2.0);
    nh.param("obstacle_clustering/size_threshold", size_threshold_, 10.0);
    nh.param("obstacle_clustering/planarity_threshold", planarity_threshold_, 0.1);
    nh.param("obstacle_clustering/min_points", min_points_, 100);
    ROS_INFO("Loaded parameters: eps=%.2f, min_samples=%d, height_threshold=%.2f, size_threshold=%.2f, planarity_threshold=%.2f, min_points=%d",
             eps_, min_samples_, height_threshold_, size_threshold_, planarity_threshold_, min_points_);
}

void ObstacleClustering::process(const pcl::PointCloud<pcl::PointXYZI>::Ptr& input_cloud,
                                 pcl::PointCloud<pcl::PointXYZI>::Ptr& output_cloud,
                                 std::vector<BoundingBox>& excluded_regions) {
    if (!input_cloud || input_cloud->empty()) {
        ROS_ERROR("Input cloud is null or empty");
        return;
    }
    ROS_INFO("Input cloud size: %zu", input_cloud->points.size());
    pcl::VoxelGrid<pcl::PointXYZI> vg;
    
    vg.setInputCloud(input_cloud);
    vg.setLeafSize(0.1f, 0.1f, 0.1f); // 体素大小可调整
    vg.filter(*input_cloud);                                
    // 添加裁剪步骤：保留边长70米的立方体内的点（x, y, z 方向各±35米）
    pcl::PointCloud<pcl::PointXYZI>::Ptr cropped_cloud(new pcl::PointCloud<pcl::PointXYZI>);
    // 使用 Eigen::aligned_allocator 定义 temp_points
    std::vector<pcl::PointXYZI, Eigen::aligned_allocator<pcl::PointXYZI>> temp_points;

    // for (const auto& point : input_cloud->points) {
    //     if (std::abs(point.x) <= 35.0 && std::abs(point.y) <= 35.0 && std::abs(point.z) <= 35.0) {
    //         cropped_cloud->points.push_back(point);
    //     }
    // }
    // cropped_cloud->width = cropped_cloud->points.size();
    // cropped_cloud->height = 1;
    // cropped_cloud->is_dense = true;

    // 并行化裁剪
    #pragma omp parallel
    {
        // 局部变量也使用相同的分配器
        std::vector<pcl::PointXYZI, Eigen::aligned_allocator<pcl::PointXYZI>> local_points;
        #pragma omp for nowait
        for (size_t i = 0; i < input_cloud->points.size(); ++i) {
            const auto& point = input_cloud->points[i];
            if (std::abs(point.x) <= 35.0 && std::abs(point.y) <= 35.0 && std::abs(point.z) <= 35.0) {
                local_points.push_back(point);
            }
        }
        #pragma omp critical
        {
            temp_points.insert(temp_points.end(), local_points.begin(), local_points.end());
        }
    }
    cropped_cloud->points = temp_points;
    cropped_cloud->width = cropped_cloud->points.size();
    cropped_cloud->height = 1;
    cropped_cloud->is_dense = true;
    ROS_INFO("Cropped cloud size: %zu", cropped_cloud->points.size());

    // 设置点云属性
    cropped_cloud->width = cropped_cloud->points.size();
    cropped_cloud->height = 1;
    cropped_cloud->is_dense = true;

    // 使用裁剪后的点云进行后续处理
    std::vector<pcl::PointIndices> cluster_indices;
    clusterPointCloud(cropped_cloud, cluster_indices);

    std::vector<bool> is_building;
    extractFeatures(cropped_cloud, cluster_indices, is_building, excluded_regions);

    filterPointCloud(cropped_cloud, is_building, cluster_indices, output_cloud);

    publishBoundingBoxes(excluded_regions);
}

void ObstacleClustering::clusterPointCloud(const pcl::PointCloud<pcl::PointXYZI>::Ptr& cloud,
                                           std::vector<pcl::PointIndices>& cluster_indices) {
    pcl::search::KdTree<pcl::PointXYZI>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZI>);
    tree->setInputCloud(cloud);

    pcl::EuclideanClusterExtraction<pcl::PointXYZI> ec;
    ec.setClusterTolerance(eps_);
    ec.setMinClusterSize(min_samples_);
    ec.setMaxClusterSize(cloud->points.size());
    ec.setSearchMethod(tree);
    ec.setInputCloud(cloud);
    ec.extract(cluster_indices);
}

void ObstacleClustering::extractFeatures(const pcl::PointCloud<pcl::PointXYZI>::Ptr& cloud,
    const std::vector<pcl::PointIndices>& cluster_indices,
    std::vector<bool>& is_building,
    std::vector<BoundingBox>& excluded_regions) {
is_building.resize(cluster_indices.size(), false);
excluded_regions.clear();

#pragma omp parallel for
for (size_t i = 0; i < cluster_indices.size(); ++i) {
pcl::PointCloud<pcl::PointXYZI>::Ptr cluster(new pcl::PointCloud<pcl::PointXYZI>);
pcl::ExtractIndices<pcl::PointXYZI> extract;
extract.setInputCloud(cloud);
extract.setIndices(boost::make_shared<pcl::PointIndices>(cluster_indices[i]));
extract.setNegative(false);
extract.filter(*cluster);

if (cluster->points.size() < min_points_) {
is_building[i] = false;
continue;
}

pcl::PointXYZI min_pt, max_pt;
pcl::getMinMax3D(*cluster, min_pt, max_pt);
double height = max_pt.z - min_pt.z;
double width = max_pt.x - min_pt.x;
double depth = max_pt.y - min_pt.y;

Eigen::Vector4f centroid;
pcl::compute3DCentroid(*cluster, centroid);

Eigen::Matrix3f covariance_matrix;
pcl::computeCovarianceMatrix(*cluster, centroid, covariance_matrix);
Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> eigen_solver(covariance_matrix);
Eigen::Vector3f eigen_values = eigen_solver.eigenvalues();
double planarity = (eigen_values[1] - eigen_values[0]) / eigen_values[2];

bool is_building_cluster = (centroid[2] < height_threshold_ || 
width > size_threshold_ || 
depth > size_threshold_ || 
planarity > planarity_threshold_);
is_building[i] = is_building_cluster;

if (is_building_cluster) {
BoundingBox bbox;
bbox.x_min = min_pt.x;
bbox.y_min = min_pt.y;
bbox.z_min = min_pt.z;
bbox.x_max = max_pt.x;
bbox.y_max = max_pt.y;
bbox.z_max = max_pt.z;
#pragma omp critical
{
excluded_regions.push_back(bbox);
}
}
}
}

void ObstacleClustering::filterPointCloud(const pcl::PointCloud<pcl::PointXYZI>::Ptr& input_cloud,
                                          const std::vector<bool>& is_building,
                                          const std::vector<pcl::PointIndices>& cluster_indices,
                                          pcl::PointCloud<pcl::PointXYZI>::Ptr& output_cloud) {
    pcl::ExtractIndices<pcl::PointXYZI> extract;
    extract.setInputCloud(input_cloud);
    extract.setNegative(true);

    pcl::PointIndices::Ptr building_indices(new pcl::PointIndices);
    for (size_t i = 0; i < cluster_indices.size(); ++i) {
        if (is_building[i]) {
            building_indices->indices.insert(building_indices->indices.end(),
                                            cluster_indices[i].indices.begin(),
                                            cluster_indices[i].indices.end());
        }
    }

    extract.setIndices(building_indices);
    extract.filter(*output_cloud);
}

void ObstacleClustering::publishBoundingBoxes(const std::vector<BoundingBox>& excluded_regions) {
    visualization_msgs::MarkerArray marker_array;
    for (size_t i = 0; i < excluded_regions.size(); ++i) {
        const BoundingBox& bbox = excluded_regions[i];
        visualization_msgs::Marker marker;
        marker.header.frame_id = "map";
        marker.header.stamp = ros::Time::now();
        marker.ns = "building_boxes";
        marker.id = i;
        marker.type = visualization_msgs::Marker::CUBE;
        marker.action = visualization_msgs::Marker::ADD;

        marker.pose.position.x = (bbox.x_min + bbox.x_max) / 2.0;
        marker.pose.position.y = (bbox.y_min + bbox.y_max) / 2.0;
        marker.pose.position.z = (bbox.z_min + bbox.z_max) / 2.0;

        marker.scale.x = bbox.x_max - bbox.x_min;
        marker.scale.y = bbox.y_max - bbox.y_min;
        marker.scale.z = bbox.z_max - bbox.z_min;

        marker.color.r = 1.0;
        marker.color.g = 0.0;
        marker.color.b = 0.0;
        marker.color.a = 0.5;

        marker.lifetime = ros::Duration();
        marker_array.markers.push_back(marker);
    }
    marker_pub_.publish(marker_array);
}
