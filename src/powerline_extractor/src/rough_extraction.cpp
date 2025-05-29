#include "rough_extraction.h"

RoughExtractor::RoughExtractor(ros::NodeHandle& nh, ros::NodeHandle& private_nh)
    : nh_(nh), private_nh_(private_nh) {
    loadParameters();
}

void RoughExtractor::loadParameters() {
    private_nh_.param<double>("linearity_threshold", linearity_threshold_, 0.7);
    ROS_INFO("Linearity threshold: %.3f", linearity_threshold_);
}

void RoughExtractor::extractPowerlinesByLinearity(
    const pcl::PointCloud<pcl::PointXYZI>::Ptr& input_cloud,
    pcl::PointCloud<pcl::PointXYZI>::Ptr& powerline_cloud) {

    powerline_cloud->clear();
    if (input_cloud->empty()) {
        return;
    }

    pcl::KdTreeFLANN<pcl::PointXYZI> kdtree;
    kdtree.setInputCloud(input_cloud);

    std::vector<int> pointIdxRadiusSearch;
    std::vector<float> pointRadiusSquaredDistance;

    for (size_t i = 0; i < input_cloud->size(); ++i) {
        if (kdtree.radiusSearch(input_cloud->points[i], 0.5, pointIdxRadiusSearch, pointRadiusSquaredDistance) < 3)
            continue;

        Eigen::MatrixXf neighborhood(pointIdxRadiusSearch.size(), 3);
        for (size_t j = 0; j < pointIdxRadiusSearch.size(); ++j) {
            neighborhood(j, 0) = input_cloud->points[pointIdxRadiusSearch[j]].x;
            neighborhood(j, 1) = input_cloud->points[pointIdxRadiusSearch[j]].y;
            neighborhood(j, 2) = input_cloud->points[pointIdxRadiusSearch[j]].z;
        }

        Eigen::MatrixXf centered = neighborhood.rowwise() - neighborhood.colwise().mean();
        Eigen::MatrixXf cov = (centered.transpose() * centered) / float(neighborhood.rows() - 1);

        Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> eig(cov);
        Eigen::Vector3f eigenvalues = eig.eigenvalues();
        Eigen::Matrix3f eigenvectors = eig.eigenvectors();

        std::vector<std::pair<float, int>> eigenvalue_indices;
        for (int j = 0; j < 3; ++j) {
            eigenvalue_indices.push_back(std::make_pair(eigenvalues(j), j));
        }
        std::sort(eigenvalue_indices.begin(), eigenvalue_indices.end(), [](const std::pair<float, int>& a, const std::pair<float, int>& b) {
            return a.first > b.first;
        });

        float lambda1 = eigenvalues(eigenvalue_indices[0].second);
        float lambda2 = eigenvalues(eigenvalue_indices[1].second);
        float lambda3 = eigenvalues(eigenvalue_indices[2].second);

        float linearity = (lambda1 - lambda2) / (lambda1 + 1e-8);

        if (linearity > linearity_threshold_) {
            powerline_cloud->push_back(input_cloud->points[i]);
        }
    }

    powerline_cloud->width = powerline_cloud->size();
    powerline_cloud->height = 1;
    powerline_cloud->is_dense = true;

    ROS_INFO("Rough Extraction: Extracted %zu powerline points", powerline_cloud->size());
}
