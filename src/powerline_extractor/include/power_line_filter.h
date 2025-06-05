#ifndef POWER_LINE_FILTER_H
#define POWER_LINE_FILTER_H

#include <ros/ros.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/kdtree/kdtree_flann.h>  // 使用KdTree进行邻近搜索

class PowerLineFilter {
public:
    // 构造函数，传入 ROS 节点句柄以加载参数
    PowerLineFilter(ros::NodeHandle& nh);

    // 过滤函数
    // 输入：粗提取的电力线点云和环境点云
    // 输出：过滤后的电力线点云
    void filter(const pcl::PointCloud<pcl::PointXYZI>::Ptr& power_line_pc,
                const pcl::PointCloud<pcl::PointXYZI>::Ptr& env_pc,
                pcl::PointCloud<pcl::PointXYZI>::Ptr& filtered_pc);

private:
    // 参数变量
    double clustering_distance_;       // 聚类距离阈值
    double length_threshold_;          // 长度过滤阈值
    double proximity_radius_;          // 邻近过滤的圆柱半径
    int proximity_count_threshold_;    // 邻近点数阈值

    // 手动实现欧几里得聚类
    void manualEuclideanClustering(const pcl::PointCloud<pcl::PointXYZI>::Ptr& cloud,
                                   std::vector<pcl::PointIndices>& cluster_indices,
                                   double tolerance, int min_size, int max_size);
};

#endif // POWER_LINE_FILTER_H
