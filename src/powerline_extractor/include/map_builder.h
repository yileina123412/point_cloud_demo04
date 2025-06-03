#ifndef MAP_BUILDER_H
#define MAP_BUILDER_H

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>
#include <ros/ros.h>
#include <map>

class MapBuilder {
public:
    // 构造函数和析构函数
    MapBuilder();
    ~MapBuilder() = default;

    // 处理输入点云，更新并输出静态和动态地图
    void processPointCloud(const pcl::PointCloud<pcl::PointXYZI>::Ptr& input_cloud,
                          pcl::PointCloud<pcl::PointXYZI>::Ptr& static_map,
                          pcl::PointCloud<pcl::PointXYZI>::Ptr& dynamic_map);

private:
    // 参数读取函数
    void loadParameters();

    // 更新静态地图
    void updateStaticMap(const pcl::PointCloud<pcl::PointXYZI>::Ptr& input_cloud);

    // 检测动态障碍物并更新动态地图
    void updateDynamicMap(const pcl::PointCloud<pcl::PointXYZI>::Ptr& input_cloud);

    // 降采样点云
    void downsampleCloud(const pcl::PointCloud<pcl::PointXYZI>::Ptr& input_cloud,
                         pcl::PointCloud<pcl::PointXYZI>::Ptr& output_cloud);

    // 成员变量
    // 参数
    double voxel_size_;               // 体素网格大小
    double confidence_increment_;     // 静态点置信度增量
    double confidence_decay_rate_;    // 未观测时的置信度衰减率
    double confidence_threshold_;     // 置信度阈值
    double static_accumulation_time_; // 静态地图初始累积时间（秒）
    double dynamic_detection_window_; // 动态检测时间窗口（秒）
    double downsample_factor_;        // 降采样因子

    // 数据结构
    pcl::PointCloud<pcl::PointXYZI>::Ptr static_map_;      // 静态地图
    pcl::PointCloud<pcl::PointXYZI>::Ptr short_term_map_;  // 短期累积点云
    std::map<int, float> voxel_confidence_;                // 体素置信度（索引->置信度）
    std::map<int, ros::Time> voxel_timestamps_;            // 体素时间戳（索引->时间）

    // PCL滤波器
    pcl::VoxelGrid<pcl::PointXYZI> voxel_grid_filter_;

    // ROS相关
    ros::NodeHandle nh_;
    ros::Time last_update_time_;  // 上次更新时间
};

#endif // MAP_BUILDER_H
