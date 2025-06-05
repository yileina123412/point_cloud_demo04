#include "power_line_filter.h"
#include <pcl/common/pca.h>
#include <pcl/common/common.h>
#include <pcl/search/kdtree.h>
#include <Eigen/Dense>
#include <limits>
#include <vector>
#include <set>

PowerLineFilter::PowerLineFilter(ros::NodeHandle& nh) {
    // 从 ROS 参数服务器加载参数，若未设置则使用默认值
    nh.param("powerline_filter/clustering_distance", clustering_distance_, 0.5);
    nh.param("powerline_filter/length_threshold", length_threshold_, 10.0);
    nh.param("powerline_filter/proximity_radius", proximity_radius_, 1.0);
    nh.param("powerline_filter/proximity_count_threshold", proximity_count_threshold_, 100);
}

// 手动实现的欧几里得聚类
void PowerLineFilter::manualEuclideanClustering(const pcl::PointCloud<pcl::PointXYZI>::Ptr& cloud,
                                                std::vector<pcl::PointIndices>& cluster_indices,
                                                double tolerance, int min_size, int max_size) {
    // 创建KdTree用于快速邻近搜索
    pcl::search::KdTree<pcl::PointXYZI>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZI>);
    tree->setInputCloud(cloud);

    std::vector<bool> processed(cloud->size(), false);
    std::vector<int> nn_indices;
    std::vector<float> nn_distances;

    for (size_t i = 0; i < cloud->size(); ++i) {
        if (processed[i]) continue;

        std::set<int> current_cluster;
        std::vector<int> seed_queue;
        seed_queue.push_back(i);
        processed[i] = true;

        size_t queue_pos = 0;
        while (queue_pos < seed_queue.size()) {
            int point_idx = seed_queue[queue_pos++];
            current_cluster.insert(point_idx);

            // 查找当前点的邻近点
            tree->radiusSearch(cloud->points[point_idx], tolerance, nn_indices, nn_distances);

            for (size_t j = 0; j < nn_indices.size(); ++j) {
                int neighbor_idx = nn_indices[j];
                if (!processed[neighbor_idx]) {
                    seed_queue.push_back(neighbor_idx);
                    processed[neighbor_idx] = true;
                }
            }
        }

        // 检查聚类大小是否在范围内
        if (current_cluster.size() >= static_cast<size_t>(min_size) &&
            current_cluster.size() <= static_cast<size_t>(max_size)) {
            pcl::PointIndices indices;
            indices.indices.assign(current_cluster.begin(), current_cluster.end());
            cluster_indices.push_back(indices);
        }
    }
}

void PowerLineFilter::filter(const pcl::PointCloud<pcl::PointXYZI>::Ptr& power_line_pc,
                             const pcl::PointCloud<pcl::PointXYZI>::Ptr& env_pc,
                             pcl::PointCloud<pcl::PointXYZI>::Ptr& filtered_pc) {
    // 清空输出点云
    filtered_pc->clear();

    // 1. 手动欧几里得聚类
    std::vector<pcl::PointIndices> cluster_indices;
    manualEuclideanClustering(power_line_pc, cluster_indices, clustering_distance_, 10, 10000);

    // 2. 对每个聚类进行过滤
    for (const auto& indices : cluster_indices) {
        // 提取当前聚类的点云
        pcl::PointCloud<pcl::PointXYZI>::Ptr cluster(new pcl::PointCloud<pcl::PointXYZI>);
        for (const auto& idx : indices.indices) {
            cluster->push_back((*power_line_pc)[idx]);
        }

        // 计算 PCA
        pcl::PCA<pcl::PointXYZI> pca;
        pca.setInputCloud(cluster);
        Eigen::Vector4f mean = pca.getMean();
        Eigen::Vector3f centroid(mean[0], mean[1], mean[2]);
        Eigen::Vector3f direction = pca.getEigenVectors().col(0);  // 主方向

        // 计算沿主方向的投影范围（长度）
        float t_min = std::numeric_limits<float>::max();
        float t_max = std::numeric_limits<float>::min();
        for (const auto& point : *cluster) {
            float t = (point.getVector3fMap() - centroid).dot(direction);
            t_min = std::min(t_min, t);
            t_max = std::max(t_max, t);
        }
        float length = t_max - t_min;

        // 长度过滤
        if (length < length_threshold_) {
            continue;  // 丢弃短聚类
        }

        // 邻近过滤
        int count = 0;
        for (const auto& env_point : *env_pc) {
            Eigen::Vector3f p = env_point.getVector3fMap();
            float t = (p - centroid).dot(direction);
            float t_clamped = std::max(t_min, std::min(t, t_max));
            Eigen::Vector3f closest = centroid + t_clamped * direction;
            float distance = (p - closest).norm();
            if (distance < proximity_radius_) {
                count++;
            }
        }
        if (count <= proximity_count_threshold_) {
            // 保留通过邻近过滤的聚类
            *filtered_pc += *cluster;
        }
    }
}
