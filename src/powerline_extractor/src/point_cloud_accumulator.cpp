#include "point_cloud_accumulator.h"
#include <pcl/filters/voxel_grid.h>


/**
 * 从ROS参数服务器加载参数
 */
void PointCloudAccumulator::loadParamsFromServer(ros::NodeHandle& nh) {
    nh.param<double>("accumulator/accumulation_time", accumulation_time_, 0.8);
    nh.param<double>("accumulator/voxel_size", voxel_size_, 0.1);
    nh.param<double>("accumulator/stability_threshold", stability_threshold_, 0.05);
    nh.param<int>("accumulator/min_observations", min_observations_, 3);
    nh.param<double>("accumulator/max_vibration_amplitude", max_vibration_amplitude_, 0.03);
    nh.param<double>("accumulator/search_radius", search_radius_, 0.8);
    nh.param<double>("accumulator/density_percentile", density_percentile_, 0.25);
    nh.param<float>("accumulator/voxel_leaf_size", voxel_leaf_size_, 0.1);


    ROS_INFO("PointCloudAccumulator parameters loaded:");
    ROS_INFO("  Accumulation time: %.1fs", accumulation_time_);
    ROS_INFO("  Voxel size: %.2fm", voxel_size_);
    ROS_INFO("  Stability threshold: %.3fm", stability_threshold_);
    ROS_INFO("  Min observations: %d", min_observations_);
    ROS_INFO("  Max vibration amplitude: %.3fm", max_vibration_amplitude_);
    ROS_INFO("  Search radius: %.2fm", search_radius_);
    ROS_INFO("  Density percentile: %.2f", density_percentile_);
    ROS_INFO("  voxel_leaf_size: %.1f", voxel_leaf_size_);
}

void PointCloudAccumulator::cleanOldClouds(const ros::Time& current_time) {
    while (!timestamp_buffer_.empty() && 
          (current_time - timestamp_buffer_.front()).toSec() > accumulation_time_) {
        cloud_buffer_.pop_front();
        timestamp_buffer_.pop_front();
    }
}

void PointCloudAccumulator::updateVoxelMap() {
    voxel_map_.clear();
    
    // 遍历所有缓冲的点云
    for (size_t i = 0; i < cloud_buffer_.size(); ++i) {
        const auto& cloud = cloud_buffer_[i];
        const auto& timestamp = timestamp_buffer_[i];
        
        for (const auto& point : cloud->points) {
            VoxelKey key = getVoxelKey(point);
            VoxelData& voxel = voxel_map_[key];
            
            voxel.points.push_back(point);
            voxel.observation_count++;
            
            if (voxel.observation_count == 1) {
                voxel.first_seen = timestamp;
            }
            voxel.last_seen = timestamp;
        }
    }
    
    // 计算每个体素的统计信息
    for (auto& pair : voxel_map_) {
        updateVoxelStatistics(pair.second);
    }
}

void PointCloudAccumulator::getAccumulateCloud()
{
    if (!accumulated_cloud) {
        accumulated_cloud.reset(new pcl::PointCloud<pcl::PointXYZI>());
    }
    // 清空 accumulated_cloud
    accumulated_cloud->clear();
    // 累积 cloud_buffer_ 中的所有点云
    for (const auto& cloud : cloud_buffer_) {
        *accumulated_cloud += *cloud;
    }
    // 使用 VoxelGrid 滤波器降采样
    pcl::VoxelGrid<pcl::PointXYZI> voxel_filter;
    voxel_filter.setInputCloud(accumulated_cloud);
    voxel_filter.setLeafSize(voxel_leaf_size_, voxel_leaf_size_, voxel_leaf_size_);

    pcl::PointCloud<pcl::PointXYZI>::Ptr downsampled_cloud(new pcl::PointCloud<pcl::PointXYZI>());
    voxel_filter.filter(*downsampled_cloud);

    // 替换 accumulated_cloud
    *accumulated_cloud = *downsampled_cloud;

}

void PointCloudAccumulator::updateVoxelStatistics(VoxelData& voxel) {
    if (voxel.points.empty()) return;
    
    // 计算平均位置
    Eigen::Vector3f sum_pos = Eigen::Vector3f::Zero();
    float sum_intensity = 0.0f;
    
    for (const auto& point : voxel.points) {
        sum_pos += Eigen::Vector3f(point.x, point.y, point.z);
        sum_intensity += point.intensity;
    }
    
    voxel.mean_position = sum_pos / voxel.points.size();
    voxel.mean_intensity = sum_intensity / voxel.points.size();
    
    // 计算位置方差(稳定性指标)
    float sum_variance = 0.0f;
    for (const auto& point : voxel.points) {
        Eigen::Vector3f pos(point.x, point.y, point.z);
        sum_variance += (pos - voxel.mean_position).squaredNorm();
    }
    voxel.position_variance = sum_variance / voxel.points.size();
}

PointCloudAccumulator::VoxelKey PointCloudAccumulator::getVoxelKey(const pcl::PointXYZI& point) const {
    return {
        static_cast<int>(std::floor(point.x / voxel_size_)),
        static_cast<int>(std::floor(point.y / voxel_size_)),
        static_cast<int>(std::floor(point.z / voxel_size_))
    };
}
//检查是否是稳定的点云：通过判断是否多次观察到有点云
bool PointCloudAccumulator::isStableVoxel(const VoxelData& voxel) const {
    return (voxel.observation_count >= min_observations_) &&
           (std::sqrt(voxel.position_variance) < stability_threshold_);
}
//检查是否抖动：通过检查体素内点云是都在某一个特征方向很大  不太好，最好换成通过前后时间判断的
bool PointCloudAccumulator::hasVibrationIssue(const VoxelData& voxel) const {
    if (voxel.points.size() < 3) return false;
    
    // 使用PCA检测抖动
    Eigen::MatrixXf points_matrix(voxel.points.size(), 3);
    for (size_t i = 0; i < voxel.points.size(); ++i) {
        points_matrix(i, 0) = voxel.points[i].x;
        points_matrix(i, 1) = voxel.points[i].y;
        points_matrix(i, 2) = voxel.points[i].z;
    }
    
    Eigen::Vector3f centroid = points_matrix.colwise().mean();
    Eigen::MatrixXf centered = points_matrix.rowwise() - centroid.transpose();
    Eigen::Matrix3f covariance = centered.transpose() * centered / (voxel.points.size() - 1);
    
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> solver(covariance);
    Eigen::Vector3f eigenvalues = solver.eigenvalues();
    
    float max_spread = std::sqrt(eigenvalues.maxCoeff());
    return max_spread > max_vibration_amplitude_;
}

pcl::PointCloud<pcl::PointXYZI>::Ptr PointCloudAccumulator::extractStablePoints() {
    pcl::PointCloud<pcl::PointXYZI>::Ptr stable_cloud(new pcl::PointCloud<pcl::PointXYZI>);
    
    for (const auto& pair : voxel_map_) {
        const VoxelData& voxel = pair.second;
        
        if (isStableVoxel(voxel) && !hasVibrationIssue(voxel)) {
            pcl::PointXYZI stable_point;
            stable_point.x = voxel.mean_position.x();
            stable_point.y = voxel.mean_position.y();
            stable_point.z = voxel.mean_position.z();
            stable_point.intensity = voxel.mean_intensity;
            
            stable_cloud->push_back(stable_point);
        }
    }
    
    stable_cloud->width = stable_cloud->size();
    stable_cloud->height = 1;
    stable_cloud->is_dense = true;
    
    return stable_cloud;
}
//密度过滤
pcl::PointCloud<pcl::PointXYZI>::Ptr PointCloudAccumulator::filterByDensity(
    const pcl::PointCloud<pcl::PointXYZI>::Ptr& input_cloud) {
    
    if (input_cloud->size() < 10) return input_cloud;
    
    pcl::PointCloud<pcl::PointXYZI>::Ptr filtered_cloud(new pcl::PointCloud<pcl::PointXYZI>);
    pcl::KdTreeFLANN<pcl::PointXYZI> kdtree;
    kdtree.setInputCloud(input_cloud);
    
    // 计算每个点的局部密度
    std::vector<float> densities;   
    densities.reserve(input_cloud->size());
    //遍历输入点云
    for (const auto& point : input_cloud->points) {
        std::vector<int> neighbor_indices;
        std::vector<float> neighbor_distances;
        
        int neighbors = kdtree.radiusSearch(point, search_radius_, 
                                          neighbor_indices, neighbor_distances);
        
        double search_volume = (4.0/3.0) * M_PI * std::pow(search_radius_, 3);
        float density = static_cast<float>(neighbors / search_volume);
        densities.push_back(density);  //每个点的在一个半径内的局部密度
    }
    
    // 计算密度阈值
    std::vector<float> sorted_densities = densities;
    std::sort(sorted_densities.begin(), sorted_densities.end());
    
    size_t threshold_index = static_cast<size_t>(sorted_densities.size() * density_percentile_);
    float density_threshold = sorted_densities[threshold_index];
    
    // 过滤低密度点
    for (size_t i = 0; i < input_cloud->size(); ++i) {
        if (densities[i] >= density_threshold) {
            filtered_cloud->push_back(input_cloud->points[i]);
        }
    }
    
    filtered_cloud->width = filtered_cloud->size();
    filtered_cloud->height = 1;
    filtered_cloud->is_dense = true;
    
    return filtered_cloud;
}

double PointCloudAccumulator::calculateAdaptiveVoxelSize(const pcl::PointCloud<pcl::PointXYZI>::Ptr& cloud) {
    if (cloud->size() < 2) return voxel_size_;
    
    pcl::KdTreeFLANN<pcl::PointXYZI> kdtree;
    kdtree.setInputCloud(cloud);
    
    double total_distance = 0.0;
    int valid_points = 0;
    
    // 随机采样计算平均最近邻距离
    size_t sample_size = std::min(cloud->size(), size_t(100));
    for (size_t i = 0; i < sample_size; ++i) {
        size_t idx = i * cloud->size() / sample_size;
        const auto& point = cloud->points[idx];
        
        std::vector<int> neighbor_indices(2);
        std::vector<float> neighbor_distances(2);
        
        if (kdtree.nearestKSearch(point, 2, neighbor_indices, neighbor_distances) >= 2) {
            total_distance += std::sqrt(neighbor_distances[1]);
            valid_points++;
        }
    }
    
    if (valid_points > 0) {
        double mean_distance = total_distance / valid_points;
        return std::max(0.05, std::min(0.2, mean_distance * 2.5));
    }
    
    return voxel_size_;
}

pcl::PointCloud<pcl::PointXYZI>::Ptr PointCloudAccumulator::adaptiveVoxelFilter(
    const pcl::PointCloud<pcl::PointXYZI>::Ptr& input_cloud) {
    
    if (input_cloud->empty()) return input_cloud;
    
    // 计算自适应体素大小
    double adaptive_voxel_size = calculateAdaptiveVoxelSize(input_cloud);
    
    // 应用体素滤波
    pcl::VoxelGrid<pcl::PointXYZI> voxel_filter;
    voxel_filter.setInputCloud(input_cloud);
    voxel_filter.setLeafSize(adaptive_voxel_size, adaptive_voxel_size, adaptive_voxel_size);
    
    pcl::PointCloud<pcl::PointXYZI>::Ptr filtered_cloud(new pcl::PointCloud<pcl::PointXYZI>);
    voxel_filter.filter(*filtered_cloud);
    
    return filtered_cloud;
}




















