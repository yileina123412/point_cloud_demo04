#include "map_builder.h"
#include <pcl/common/transforms.h>
#include <pcl_conversions/pcl_conversions.h>

MapBuilder::MapBuilder() : nh_("~") {
    // 初始化点云
    static_map_.reset(new pcl::PointCloud<pcl::PointXYZI>());
    short_term_map_.reset(new pcl::PointCloud<pcl::PointXYZI>());

    // 加载参数
    loadParameters();

    // 初始化体素网格滤波器
    voxel_grid_filter_.setLeafSize(voxel_size_, voxel_size_, voxel_size_);

    // 设置初始时间
    last_update_time_ = ros::Time::now();
}

void MapBuilder::loadParameters() {
    // 从ROS参数服务器读取参数
    nh_.param("map_bulid/voxel_size", voxel_size_, 0.1);                    // 默认0.1m
    nh_.param("map_bulid/confidence_increment", confidence_increment_, 0.1); // 默认增量0.1
    nh_.param("map_bulid/confidence_decay_rate", confidence_decay_rate_, 0.05); // 默认衰减率0.05/秒
    nh_.param("map_bulid/confidence_threshold", confidence_threshold_, 0.2); // 默认阈值0.2
    nh_.param("map_bulid/static_accumulation_time", static_accumulation_time_, 10.0); // 默认10秒
    nh_.param("map_bulid/dynamic_detection_window", dynamic_detection_window_, 1.0);  // 默认1秒
    nh_.param("map_bulid/downsample_factor", downsample_factor_, 0.5);      // 默认降采样50%

    ROS_INFO("Map build Parameters loaded: voxel_size=%.2f, confidence_increment=%.2f, confidence_decay_rate=%.2f",
             voxel_size_, confidence_increment_, confidence_decay_rate_);
}

void MapBuilder::processPointCloud(const pcl::PointCloud<pcl::PointXYZI>::Ptr& input_cloud,
                                   pcl::PointCloud<pcl::PointXYZI>::Ptr& static_map,
                                   pcl::PointCloud<pcl::PointXYZI>::Ptr& dynamic_map) {
    // 记录开始时间
    ros::Time start_time = ros::Time::now();

    // 更新静态地图
    updateStaticMap(input_cloud);

    // 更新动态地图
    // updateDynamicMap(input_cloud);

    pcl::PointCloud<pcl::PointXYZI>::Ptr filtered_static_map(new pcl::PointCloud<pcl::PointXYZI>());
    for (const auto& point : static_map_->points) {
        int idx = static_cast<int>(point.x / voxel_size_) * 10000 +
                static_cast<int>(point.y / voxel_size_) * 100 +
                static_cast<int>(point.z / voxel_size_);
        if (voxel_confidence_.count(idx) > 0 && voxel_confidence_[idx] >= confidence_threshold_) {
            filtered_static_map->points.push_back(point);
        }
    }

    // 输出结果
    *static_map = *static_map_;
    // *dynamic_map = *short_term_map_; // 动态地图直接使用短期累积结果，后续可聚类
    // 记录结束时间并计算耗时
    ros::Time end_time = ros::Time::now();
    double processing_time = (end_time - start_time).toSec();
    ROS_INFO("构建静态地图耗时: %.4f seconds", processing_time);
    ROS_INFO("静态地图点云数量:%ld",static_map->size());
}

void MapBuilder::updateStaticMap(const pcl::PointCloud<pcl::PointXYZI>::Ptr& input_cloud) {
    ros::Time current_time = ros::Time::now();
    double dt = (current_time - last_update_time_).toSec();

    // 降采样输入点云
    pcl::PointCloud<pcl::PointXYZI>::Ptr filtered_cloud(new pcl::PointCloud<pcl::PointXYZI>());
    downsampleCloud(input_cloud, filtered_cloud);

    // 更新体素网格和置信度
    voxel_grid_filter_.setInputCloud(filtered_cloud);
    voxel_grid_filter_.filter(*filtered_cloud);

    for (const auto& point : filtered_cloud->points) {
        // 计算体素索引（简化为整数网格坐标）
        int idx = static_cast<int>(point.x / voxel_size_) * 10000 +
                  static_cast<int>(point.y / voxel_size_) * 100 +
                  static_cast<int>(point.z / voxel_size_);

        // 更新置信度
        if (voxel_confidence_.find(idx) == voxel_confidence_.end()) {
            voxel_confidence_[idx] = confidence_increment_;
        } else {
            voxel_confidence_[idx] += confidence_increment_;
            if (voxel_confidence_[idx] > 1.0) voxel_confidence_[idx] = 1.0; // 上限
        }
        voxel_timestamps_[idx] = current_time;

        // 添加到静态地图
        static_map_->points.push_back(point);
    }
    


    // 滑动窗口：限制点云数量（例如最多保留 10000 个点）
    // const size_t max_points = 50000;
    // if (static_map_->points.size() > max_points) {
    //     static_map_->points.erase(static_map_->points.begin(), 
    //                               static_map_->points.begin() + (static_map_->points.size() - max_points));
    // }

    // 衰减未观测点的置信度
    for (auto it = voxel_confidence_.begin(); it != voxel_confidence_.end();) {
        double time_diff = (current_time - voxel_timestamps_[it->first]).toSec();
        it->second -= confidence_decay_rate_ * time_diff;
        if (it->second < confidence_threshold_) {
            it = voxel_confidence_.erase(it);
        } else {
            ++it;
 
        }
    }

    // 更新时间
    last_update_time_ = current_time;

    // 降采样静态地图
    downsampleCloud(static_map_, static_map_);
}

void MapBuilder::updateDynamicMap(const pcl::PointCloud<pcl::PointXYZI>::Ptr& input_cloud) {
    short_term_map_->clear();
    pcl::PointCloud<pcl::PointXYZI>::Ptr filtered_cloud(new pcl::PointCloud<pcl::PointXYZI>());
    downsampleCloud(input_cloud, filtered_cloud);

    for (const auto& point : filtered_cloud->points) {
        int idx = static_cast<int>(point.x / voxel_size_) * 10000 +
                  static_cast<int>(point.y / voxel_size_) * 100 +
                  static_cast<int>(point.z / voxel_size_);

        // 仅当点不在静态地图中且置信度未积累到一定程度时，视为动态点
        if (voxel_confidence_.find(idx) == voxel_confidence_.end() ||
            voxel_confidence_[idx] < confidence_threshold_) {
            // 额外检查：如果点在静态地图中已有一定观测历史，则不视为动态点
            if (voxel_timestamps_.find(idx) == voxel_timestamps_.end() ||
                (ros::Time::now() - voxel_timestamps_[idx]).toSec() < static_accumulation_time_) {
                short_term_map_->points.push_back(point);
            }
        }
    }
}


void MapBuilder::downsampleCloud(const pcl::PointCloud<pcl::PointXYZI>::Ptr& input_cloud,
                                 pcl::PointCloud<pcl::PointXYZI>::Ptr& output_cloud) {
    voxel_grid_filter_.setInputCloud(input_cloud);
    voxel_grid_filter_.setLeafSize(downsample_factor_ * voxel_size_,
                                   downsample_factor_ * voxel_size_,
                                   downsample_factor_ * voxel_size_);
    voxel_grid_filter_.filter(*output_cloud);
}
