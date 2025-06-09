#include "power_line_coarse_extractor_s.h"
#include <Eigen/Dense>
#include <queue>

PowerLineExtractor::PowerLineExtractor(ros::NodeHandle& nh) : extracted_cloud_(new pcl::PointCloud<pcl::PointXYZI>) {
    loadParameters(nh);
    normal_estimation_.setRadiusSearch(0.5);
    kdtree_.reset(new pcl::search::KdTree<pcl::PointXYZI>);
}

void PowerLineExtractor::loadParameters(ros::NodeHandle& nh) {
    // ros::NodeHandle nh("~");
    nh.param("power_line_coarse_extractor_s/linearity_threshold", linearity_threshold_, 0.7);
    nh.param("power_line_coarse_extractor_s/curvature_threshold", curvature_threshold_, 0.1);
    nh.param("power_line_coarse_extractor_s/planarity_threshold", planarity_threshold_, 0.1);
    nh.param("power_line_coarse_extractor_s/use_planarity", use_planarity_, false);
    nh.param("power_line_coarse_extractor_s/cluster_tolerance", cluster_tolerance_, 0.5);
    nh.param("power_line_coarse_extractor_s/min_cluster_size", min_cluster_size_, 10);
    // 新增参数
    nh.param("power_line_coarse_extractor_s/variance_threshold", variance_threshold_, 0.1);
    nh.param("power_line_coarse_extractor_s/search_radius", search_radius_, 0.5);
    // 新增参数
    nh.param("power_line_coarse_extractor_s/min_cluster_length", min_cluster_length_, 5.0); // 单位：米
    ROS_INFO("粗提取_s参数加载完成");
    ROS_INFO("min_cluster_length: %.2f",min_cluster_length_);
}

void PowerLineExtractor::extractPowerLines(const std::unique_ptr<PointCloudPreprocessor>& preprocessor) {
    auto cloud = preprocessor->getProcessedCloud();
    ROS_INFO("preprocessor->getProcessedCloud() 粗提取_s: %ld",cloud->size());
    auto& octree = preprocessor->getOctree();
    kdtree_->setInputCloud(cloud);

    pcl::PointCloud<pcl::PointXYZI>::Ptr power_lines(new pcl::PointCloud<pcl::PointXYZI>);

    // 获取 Octree 叶节点中心点
    std::vector<pcl::PointXYZI, Eigen::aligned_allocator<pcl::PointXYZI>> voxel_centers;
    octree.getOccupiedVoxelCenters(voxel_centers);

    // 遍历每个体素中心
    for (const auto& center : voxel_centers) {
        std::vector<int> point_indices;
        std::vector<float> distances;
        kdtree_->radiusSearch(center, octree.getResolution() / 2.0, point_indices, distances);

        if (point_indices.empty()) continue;

        pcl::PointCloud<pcl::PointXYZI>::Ptr leaf_cloud(new pcl::PointCloud<pcl::PointXYZI>);
        pcl::copyPointCloud(*cloud, point_indices, *leaf_cloud);

        // 手动计算 PCA
        Eigen::Vector4f centroid;
        pcl::compute3DCentroid(*leaf_cloud, centroid);

        Eigen::Matrix3f covariance;
        pcl::computeCovarianceMatrix(*leaf_cloud, centroid, covariance);
        ROS_INFO("Leaf cloud size: %zu", leaf_cloud->points.size());

        Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> eigen_solver(covariance);
        Eigen::Vector3f eigenvalues = eigen_solver.eigenvalues();
        std::sort(eigenvalues.data(), eigenvalues.data() + 3, std::greater<float>());

        float linearity = (eigenvalues[0] - eigenvalues[1]) / eigenvalues[0];
        float curvature = eigenvalues[2] / (eigenvalues[0] + eigenvalues[1] + eigenvalues[2]);
        float planarity = (eigenvalues[1] - eigenvalues[2]) / eigenvalues[0];

        // 计算法向量一致性
        pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>);
        normal_estimation_.setInputCloud(leaf_cloud);
        normal_estimation_.compute(*normals);
        Eigen::Vector3f mean_normal(0, 0, 0);
        for (const auto& normal : normals->points) {
            mean_normal += Eigen::Vector3f(normal.normal_x, normal.normal_y, normal.normal_z);
        }
        mean_normal /= normals->size();
        float variance = 0;
        for (const auto& normal : normals->points) {
            Eigen::Vector3f diff = Eigen::Vector3f(normal.normal_x, normal.normal_y, normal.normal_z) - mean_normal;
            variance += diff.squaredNorm();
        }
        variance /= normals->size();

        // 筛选条件
        bool is_power_line = linearity > linearity_threshold_ && curvature < curvature_threshold_;

        if (use_planarity_) {
            is_power_line = is_power_line && planarity < planarity_threshold_;
        }
        is_power_line = is_power_line && variance > 0.1; // 排除面状结构

        if (is_power_line) {
            ROS_INFO("满足");

            *power_lines += *leaf_cloud;
        }

    }

    // 手动聚类
    std::vector<pcl::PointIndices> cluster_indices;
    manualClustering(power_lines, cluster_indices, cluster_tolerance_, min_cluster_size_);

    // 提取聚类结果
    extracted_cloud_->clear();
    for (const auto& indices : cluster_indices) {
        pcl::PointCloud<pcl::PointXYZI>::Ptr cluster(new pcl::PointCloud<pcl::PointXYZI>);
        pcl::copyPointCloud(*power_lines, indices, *cluster);
        *extracted_cloud_ += *cluster;
    }
}

void PowerLineExtractor::manualClustering(const pcl::PointCloud<pcl::PointXYZI>::Ptr& cloud,
                                          std::vector<pcl::PointIndices>& cluster_indices,
                                          double tolerance, int min_size) {
    if (cloud->empty()) return;

    std::vector<bool> processed(cloud->size(), false);
    kdtree_->setInputCloud(cloud);

    for (size_t i = 0; i < cloud->size(); ++i) {
        if (processed[i]) continue;

        std::queue<int> queue;
        queue.push(i);
        processed[i] = true;
        pcl::PointIndices cluster;
        cluster.indices.push_back(i);

        while (!queue.empty()) {
            int idx = queue.front();
            queue.pop();

            std::vector<int> neighbors;
            std::vector<float> distances;
            kdtree_->radiusSearch(idx, tolerance, neighbors, distances);

            for (int neighbor : neighbors) {
                if (!processed[neighbor]) {
                    processed[neighbor] = true;
                    queue.push(neighbor);
                    cluster.indices.push_back(neighbor);
                }
            }
        }

        if (cluster.indices.size() >= static_cast<size_t>(min_size)) {
            cluster_indices.push_back(cluster);
        }
    }
}

pcl::PointCloud<pcl::PointXYZI>::Ptr PowerLineExtractor::getExtractedCloud() const {
    return extracted_cloud_;
}
// 新增逐点提取函数
void PowerLineExtractor::extractPowerLinesByPoints(const std::unique_ptr<PointCloudPreprocessor>& preprocessor_ptr) {
    // 获取预处理后的点云
    auto cloud = preprocessor_ptr->getProcessedCloud();
    if (cloud->empty()) {
        ROS_WARN("Input point cloud is empty!");
        return;
    }
    kdtree_->setInputCloud(cloud);

    // 计算所有点的法向量
    pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>);
    normal_estimation_.setInputCloud(cloud);
    normal_estimation_.setSearchMethod(kdtree_);
    normal_estimation_.setRadiusSearch(search_radius_);
    normal_estimation_.compute(*normals);

    // 遍历每个点，筛选电力线点
    pcl::PointCloud<pcl::PointXYZI>::Ptr power_lines(new pcl::PointCloud<pcl::PointXYZI>);
    for (size_t i = 0; i < cloud->points.size(); ++i) {
        if (isPowerLinePoint(cloud, normals, i)) {
            power_lines->points.push_back(cloud->points[i]);
        }
    }
    power_lines->width = power_lines->points.size();
    power_lines->height = 1;
    power_lines->is_dense = true;

    // 手动聚类
    std::vector<pcl::PointIndices> cluster_indices;
    manualClustering(power_lines, cluster_indices, cluster_tolerance_, min_cluster_size_);

    // 滤除较短的簇
    filterShortClusters(power_lines, cluster_indices, min_cluster_length_);

    // 提取聚类结果
    extracted_cloud_->clear();
    for (const auto& indices : cluster_indices) {
        pcl::PointCloud<pcl::PointXYZI>::Ptr cluster(new pcl::PointCloud<pcl::PointXYZI>);
        for (int idx : indices.indices) {
            cluster->points.push_back(power_lines->points[idx]);
        }
        cluster->width = cluster->points.size();
        cluster->height = 1;
        cluster->is_dense = true;
        *extracted_cloud_ += *cluster;
    }
}

// 新增辅助函数：判断单个点是否为电力线点
bool PowerLineExtractor::isPowerLinePoint(const pcl::PointCloud<pcl::PointXYZI>::Ptr& cloud,
                                          const pcl::PointCloud<pcl::Normal>::Ptr& normals,
                                          int index) {
    // 获取邻域点
    std::vector<int> neighbors;
    std::vector<float> distances;
    kdtree_->radiusSearch(index, search_radius_, neighbors, distances);

    if (neighbors.size() < 3) return false; // 邻域点数不足

    // 提取邻域点云
    pcl::PointCloud<pcl::PointXYZI>::Ptr local_cloud(new pcl::PointCloud<pcl::PointXYZI>);
    for (int idx : neighbors) {
        local_cloud->points.push_back(cloud->points[idx]);
    }

    // 手动计算 PCA
    Eigen::Vector4f centroid;
    pcl::compute3DCentroid(*local_cloud, centroid);
    Eigen::Matrix3f covariance;
    pcl::computeCovarianceMatrix(*local_cloud, centroid, covariance);
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> eigen_solver(covariance);
    Eigen::Vector3f eigenvalues = eigen_solver.eigenvalues();
    std::sort(eigenvalues.data(), eigenvalues.data() + 3, std::greater<float>());

    if (eigenvalues[0] < 1e-6) return false; // 避免除以零

    float linearity = (eigenvalues[0] - eigenvalues[1]) / eigenvalues[0];
    float curvature = eigenvalues[2] / (eigenvalues[0] + eigenvalues[1] + eigenvalues[2]);

    // 计算法向量一致性
    Eigen::Vector3f mean_normal(0, 0, 0);
    for (int idx : neighbors) {
        mean_normal += Eigen::Vector3f(normals->points[idx].normal_x,
                                       normals->points[idx].normal_y,
                                       normals->points[idx].normal_z);
    }
    mean_normal /= neighbors.size();
    float variance = 0;
    for (int idx : neighbors) {
        Eigen::Vector3f diff = Eigen::Vector3f(normals->points[idx].normal_x,
                                               normals->points[idx].normal_y,
                                               normals->points[idx].normal_z) - mean_normal;
        variance += diff.squaredNorm();
    }
    variance /= neighbors.size();

    // 筛选条件
    return linearity > linearity_threshold_ && curvature < curvature_threshold_ && variance > variance_threshold_;
}

void PowerLineExtractor::filterShortClusters(const pcl::PointCloud<pcl::PointXYZI>::Ptr& cloud,
    std::vector<pcl::PointIndices>& cluster_indices,
    double min_length) {
std::vector<pcl::PointIndices> filtered_indices;
for (const auto& cluster : cluster_indices) {
if (cluster.indices.size() < 2) continue; // 跳过点数不足的簇

// 计算簇的边界框
pcl::PointXYZI min_pt, max_pt;
min_pt.x = min_pt.y = min_pt.z = std::numeric_limits<float>::max();
max_pt.x = max_pt.y = max_pt.z = -std::numeric_limits<float>::max();
for (int idx : cluster.indices) {
const auto& point = cloud->points[idx];
min_pt.x = std::min(min_pt.x, point.x);
min_pt.y = std::min(min_pt.y, point.y);
min_pt.z = std::min(min_pt.z, point.z);
max_pt.x = std::max(max_pt.x, point.x);
max_pt.y = std::max(max_pt.y, point.y);
max_pt.z = std::max(max_pt.z, point.z);
}
// 计算边界框对角线长度
double length = std::sqrt(std::pow(max_pt.x - min_pt.x, 2) +
std::pow(max_pt.y - min_pt.y, 2) +
std::pow(max_pt.z - min_pt.z, 2));
if (length >= min_length) {
filtered_indices.push_back(cluster);
}
}
cluster_indices = filtered_indices;
}