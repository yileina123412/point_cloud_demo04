#ifndef POINT_CLOUD_ACCUMULATOR_H
#define POINT_CLOUD_ACCUMULATOR_H

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/filters/voxel_grid.h>
#include <ros/ros.h>
#include <deque>
#include <unordered_map>
#include <Eigen/Dense>

/**
 * @brief 针对固定雷达的点云累积器
 * 专门处理雷达抖动和密度不均匀问题
 */
class PointCloudAccumulator {
public:
    struct VoxelKey {
        int x, y, z;
        
        bool operator==(const VoxelKey& other) const {
            return x == other.x && y == other.y && z == other.z;
        }
    };
    
    struct VoxelKeyHash {
        std::size_t operator()(const VoxelKey& k) const {
            return std::hash<int>()(k.x) ^ (std::hash<int>()(k.y) << 1) ^ (std::hash<int>()(k.z) << 2);
        }
    };
    
    struct VoxelData {
        std::vector<pcl::PointXYZI> points;     // 体素内所有点
        int observation_count;                   // 观测次数
        Eigen::Vector3f mean_position;          // 平均位置
        float position_variance;                 // 位置方差(稳定性指标)
        float mean_intensity;                    // 平均强度
        ros::Time first_seen;                   // 首次观测时间
        ros::Time last_seen;                    // 最后观测时间
        
        VoxelData() : observation_count(0), position_variance(0.0), mean_intensity(0.0) {}
    };

public:
    PointCloudAccumulator(double accumulation_time = 0.8, 
                         double voxel_size = 0.1,
                         double stability_threshold = 0.05,
                         int min_observations = 3)
        : accumulation_time_(accumulation_time),
          voxel_size_(voxel_size),
          stability_threshold_(stability_threshold),
          min_observations_(min_observations),
          max_vibration_amplitude_(0.03),
          search_radius_(0.8),
          density_percentile_(0.25),
          voxel_leaf_size_(0.1) {
        
        ROS_INFO("PointCloudAccumulator initialized:");
        ROS_INFO("  Accumulation time: %.1fs", accumulation_time_);
        ROS_INFO("  Voxel size: %.2fm", voxel_size_);
        ROS_INFO("  Stability threshold: %.3fm", stability_threshold_);
        ROS_INFO("  Min observations: %d", min_observations_);
    }
    
    /**
     * @brief 从ROS参数服务器加载参数
     */
    void loadParamsFromServer(ros::NodeHandle& nh);
    
    /**
     * @brief 添加新的点云到累积缓冲区
     * 添加每一帧的点云，更新体素并进行时间窗口滑动
     */
    void addPointCloud(const pcl::PointCloud<pcl::PointXYZI>::Ptr& cloud, 
                       const ros::Time& timestamp) {
        // 添加到缓冲区
        cloud_buffer_.push_back(cloud);
        timestamp_buffer_.push_back(timestamp);
        
        // 移除超出时间窗口的点云
        cleanOldClouds(timestamp);
        
        // 更新体素映射
        updateVoxelMap();
        //累积点云
        getAccumulateCloud();
    }
    
    /**
     * @brief 获取累积并稳定的点云
     */
    pcl::PointCloud<pcl::PointXYZI>::Ptr getStableCloud() {
        pcl::PointCloud<pcl::PointXYZI>::Ptr stable_cloud(new pcl::PointCloud<pcl::PointXYZI>);
        
        // Step 1: 获取稳定点
        auto candidate_cloud = extractStablePoints();
        if (candidate_cloud->empty()) return stable_cloud;
        
        // Step 2: 密度过滤
        auto density_filtered = filterByDensity(candidate_cloud);
        if (density_filtered->empty()) return stable_cloud;
        
        // Step 3: 自适应体素化
        auto final_cloud = adaptiveVoxelFilter(density_filtered);
        
        return accumulated_cloud;
        // return final_cloud;
    }
    
    /**
     * @brief 获取累积统计信息
     */
    void getAccumulationStats(size_t& total_voxels, size_t& stable_voxels, 
                             double& avg_observations, double& avg_stability) const {
        total_voxels = voxel_map_.size();
        stable_voxels = 0;
        double total_obs = 0;
        double total_stability = 0;
        
        for (const auto& pair : voxel_map_) {
            const VoxelData& voxel = pair.second;
            total_obs += voxel.observation_count;
            total_stability += voxel.position_variance;
            
            if (isStableVoxel(voxel)) {
                stable_voxels++;
            }
        }
        
        avg_observations = total_voxels > 0 ? total_obs / total_voxels : 0;
        avg_stability = total_voxels > 0 ? total_stability / total_voxels : 0;
    }

private:
    // 核心方法
    void cleanOldClouds(const ros::Time& current_time);
    void updateVoxelMap();
    void updateVoxelStatistics(VoxelData& voxel);
    VoxelKey getVoxelKey(const pcl::PointXYZI& point) const;
    bool isStableVoxel(const VoxelData& voxel) const;
    bool hasVibrationIssue(const VoxelData& voxel) const;
    pcl::PointCloud<pcl::PointXYZI>::Ptr extractStablePoints();
    pcl::PointCloud<pcl::PointXYZI>::Ptr filterByDensity(const pcl::PointCloud<pcl::PointXYZI>::Ptr& input_cloud);
    double calculateAdaptiveVoxelSize(const pcl::PointCloud<pcl::PointXYZI>::Ptr& cloud);
    pcl::PointCloud<pcl::PointXYZI>::Ptr adaptiveVoxelFilter(const pcl::PointCloud<pcl::PointXYZI>::Ptr& input_cloud);

    void getAccumulateCloud();
    
    // 累积参数
    double accumulation_time_;        // 累积时间窗口 (0.8-1.0秒)
    double voxel_size_;              // 累积体素大小 (0.1米)
    double stability_threshold_;      // 稳定性阈值 (0.05米)
    int min_observations_;           // 最小观测次数 (3-5次)
    double max_vibration_amplitude_; // 最大抖动幅度 (0.03米)
    
    // 密度过滤参数
    double search_radius_;           // 密度计算半径 (0.8米)
    double density_percentile_;      // 密度过滤百分位数 (0.25)

    //降采样参数
    float voxel_leaf_size_;
    
    // 累积缓冲区
    std::deque<pcl::PointCloud<pcl::PointXYZI>::Ptr> cloud_buffer_;
    std::deque<ros::Time> timestamp_buffer_;
    
    // 空间体素映射
    std::unordered_map<VoxelKey, VoxelData, VoxelKeyHash> voxel_map_;

    pcl::PointCloud<pcl::PointXYZI>::Ptr accumulated_cloud;
};

#endif // POINT_CLOUD_ACCUMULATOR_H











