#ifndef RADAR_POWERLINE_PREPROCESSOR_H
#define RADAR_POWERLINE_PREPROCESSOR_H

#include <ros/ros.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <deque>
#include <unordered_map>
#include <Eigen/Dense>

class RadarPowerlinePreprocessor {
public:
    RadarPowerlinePreprocessor();
    ~RadarPowerlinePreprocessor();

    // 初始化函数，使用ROS节点句柄读取参数
    void initialize(ros::NodeHandle& nh);
    
    // 添加新的点云
    void addPointCloud(const pcl::PointCloud<pcl::PointXYZI>::Ptr& cloud, const ros::Time& timestamp);
    
    // 获取稳定点云
    pcl::PointCloud<pcl::PointXYZI>::Ptr getStableCloud();

private:
    // 窗口数据结构
    struct WindowData {
        ros::Time start_time;
        ros::Time end_time;
        pcl::PointCloud<pcl::PointXYZI>::Ptr accumulated_cloud;
    };

    // 体素数据结构
    struct VoxelData {
        std::vector<pcl::PointXYZI> points;
        std::vector<float> weights;
        std::vector<ros::Time> timestamps;
        ros::Time first_seen;
        ros::Time last_seen;
        std::map<ros::Time, Eigen::Vector3f> window_centroids;
        std::map<ros::Time, int> window_point_counts;
        std::map<ros::Time, bool> window_complete;
        bool is_linear;
    };

    // 哈希函数，用于体素索引
    struct VoxelIndex {
        int x, y, z;
        
        bool operator==(const VoxelIndex& other) const {
            return x == other.x && y == other.y && z == other.z;
        }
    };

    struct VoxelIndexHash {
        std::size_t operator()(const VoxelIndex& idx) const {
            return std::hash<int>()(idx.x) ^ 
                   std::hash<int>()(idx.y) ^ 
                   std::hash<int>()(idx.z);
        }
    };

    // 计算点所在的体素索引
    VoxelIndex computeVoxelIndex(const pcl::PointXYZI& point) const;
    
    // 更新体素图
    void updateVoxelMap();
    
    // 清理旧窗口
    void cleanOldWindows();
    
    // 判断体素是否有抖动问题
    bool hasVibrationIssue(const VoxelData& voxel) const;
    
    // 提取稳定点
    pcl::PointCloud<pcl::PointXYZI>::Ptr extractStablePoints();
    
    // 更新体素的线性特征
    void updateVoxelLinearity(VoxelData& voxel);

    // 窗口缓冲区
    std::deque<WindowData> window_buffer_;
    
    // 体素图，用于存储体素数据
    std::unordered_map<VoxelIndex, VoxelData, VoxelIndexHash> voxel_map_;
    
    // 当前窗口
    WindowData current_window_;
    
    // 参数
    double accumulation_time_;          // 窗口累积时长(秒)
    double max_history_time_;           // 历史窗口保留时间(秒)
    float weight_threshold_;            // 权重阈值，用于清理低权重点
    int max_points_per_voxel_;          // 每个体素的最大点数
    float voxel_size_;                  // 体素大小(米)
    int min_points_per_window_;         // 判断窗口完整性的最小点数
    int min_windows_for_vibration_;     // 延迟抖动检测的最小窗口数
    double min_accumulation_time_;      // 初期累积时间(秒)
    float max_vibration_amplitude_;     // 最大抖动幅度(米)
    float stability_threshold_;         // 稳定性阈值(米)
    int min_observations_;              // 最小观测次数
};

#endif // RADAR_POWERLINE_PREPROCESSOR_H
