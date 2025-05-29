#include "powerline_coarse_extractor.h"
#include <iostream>
#include <algorithm>
#include <cmath>

PowerlineCoarseExtractor::PowerlineCoarseExtractor() 
    : pca_radius_(0.8),
      linearity_threshold_(0.7),
      voxel_size_(0.1),
      min_range_(0.5),
      max_range_(100.0),
      min_height_(-5.0),
      max_height_(50.0),
      outlier_mean_k_(50),
      outlier_std_thresh_(1.0),
      preprocessed_cloud_(new pcl::PointCloud<pcl::PointXYZI>()),
      input_size_(0),
      preprocessed_size_(0),
      output_size_(0),
      kdtree_(new pcl::KdTreeFLANN<pcl::PointXYZI>()) {
    
    std::cout << "PowerlineCoarseExtractor initialized with default parameters:" << std::endl;
    std::cout << "  PCA radius: " << pca_radius_ << std::endl;
    std::cout << "  Linearity threshold: " << linearity_threshold_ << std::endl;
    std::cout << "  Voxel size: " << voxel_size_ << std::endl;
}

PowerlineCoarseExtractor::~PowerlineCoarseExtractor() {
    std::cout << "PowerlineCoarseExtractor destructor called" << std::endl;
}

void PowerlineCoarseExtractor::setPCARadius(double radius) {
    if (radius > 0) {
        pca_radius_ = radius;
        std::cout << "PCA radius set to: " << pca_radius_ << std::endl;
    } else {
        std::cerr << "Warning: Invalid PCA radius, keeping current value: " << pca_radius_ << std::endl;
    }
}

void PowerlineCoarseExtractor::setLinearityThreshold(double threshold) {
    if (threshold >= 0.0 && threshold <= 1.0) {
        linearity_threshold_ = threshold;
        std::cout << "Linearity threshold set to: " << linearity_threshold_ << std::endl;
    } else {
        std::cerr << "Warning: Linearity threshold should be in [0,1], keeping current value: " 
                  << linearity_threshold_ << std::endl;
    }
}

void PowerlineCoarseExtractor::setVoxelSize(double leaf_size) {
    if (leaf_size > 0) {
        voxel_size_ = leaf_size;
        std::cout << "Voxel size set to: " << voxel_size_ << std::endl;
    } else {
        std::cerr << "Warning: Invalid voxel size, keeping current value: " << voxel_size_ << std::endl;
    }
}

void PowerlineCoarseExtractor::setRangeFilter(double min_range, double max_range) {
    if (min_range >= 0 && max_range > min_range) {
        min_range_ = min_range;
        max_range_ = max_range;
        std::cout << "Range filter set to: [" << min_range_ << ", " << max_range_ << "]" << std::endl;
    } else {
        std::cerr << "Warning: Invalid range parameters, keeping current values: [" 
                  << min_range_ << ", " << max_range_ << "]" << std::endl;
    }
}

void PowerlineCoarseExtractor::setHeightFilter(double min_height, double max_height) {
    if (max_height > min_height) {
        min_height_ = min_height;
        max_height_ = max_height;
        std::cout << "Height filter set to: [" << min_height_ << ", " << max_height_ << "]" << std::endl;
    } else {
        std::cerr << "Warning: Invalid height parameters, keeping current values: [" 
                  << min_height_ << ", " << max_height_ << "]" << std::endl;
    }
}

void PowerlineCoarseExtractor::setOutlierRemovalParams(int mean_k, double std_thresh) {
    if (mean_k > 0 && std_thresh > 0) {
        outlier_mean_k_ = mean_k;
        outlier_std_thresh_ = std_thresh;
        std::cout << "Outlier removal parameters set to: mean_k=" << outlier_mean_k_ 
                  << ", std_thresh=" << outlier_std_thresh_ << std::endl;
    } else {
        std::cerr << "Warning: Invalid outlier removal parameters, keeping current values: mean_k=" 
                  << outlier_mean_k_ << ", std_thresh=" << outlier_std_thresh_ << std::endl;
    }
}

bool PowerlineCoarseExtractor::extractPowerlines(const pcl::PointCloud<pcl::PointXYZI>::Ptr& input_cloud,
                                               pcl::PointCloud<pcl::PointXYZI>::Ptr& output_cloud) {
    if (!input_cloud || input_cloud->empty()) {
        std::cerr << "Error: Input point cloud is empty or null" << std::endl;
        return false;
    }
    
    input_size_ = input_cloud->size();
    
    try {
        // 预处理点云
        preprocessPointCloud(input_cloud, preprocessed_cloud_);
        preprocessed_size_ = preprocessed_cloud_->size();
        
        if (preprocessed_cloud_->empty()) {
            std::cerr << "Warning: Preprocessed point cloud is empty" << std::endl;
            output_cloud->clear();
            output_size_ = 0;
            return true;
        }
        
        // 基于线性度提取电力线
        extractByLinearity(preprocessed_cloud_, output_cloud);
        output_size_ = output_cloud->size();
        
        std::cout << "Coarse extraction completed: " << input_size_ << " -> " 
                  << preprocessed_size_ << " -> " << output_size_ << " points" << std::endl;
        
        return true;
        
    } catch (const std::exception& e) {
        std::cerr << "Error during powerline extraction: " << e.what() << std::endl;
        return false;
    }
}

pcl::PointCloud<pcl::PointXYZI>::Ptr PowerlineCoarseExtractor::getPreprocessedCloud() const {
    return preprocessed_cloud_;
}

void PowerlineCoarseExtractor::getExtractionStats(size_t& input_size, size_t& preprocessed_size, size_t& output_size) const {
    input_size = input_size_;
    preprocessed_size = preprocessed_size_;
    output_size = output_size_;
}

void PowerlineCoarseExtractor::preprocessPointCloud(const pcl::PointCloud<pcl::PointXYZI>::Ptr& input_cloud,
                                                  pcl::PointCloud<pcl::PointXYZI>::Ptr& output_cloud) {
    // 距离和高度过滤
    pcl::PointCloud<pcl::PointXYZI>::Ptr range_filtered(new pcl::PointCloud<pcl::PointXYZI>);
    filterPointCloudByRange(input_cloud, range_filtered);
    
    // 体素下采样
    pcl::PointCloud<pcl::PointXYZI>::Ptr downsampled(new pcl::PointCloud<pcl::PointXYZI>);
    downsamplePointCloud(range_filtered, downsampled);
    
    // 移除离群点
    removeOutliers(downsampled, output_cloud);
    
    std::cout << "Preprocessing: " << input_cloud->size() << " -> " 
              << range_filtered->size() << " -> " << downsampled->size() 
              << " -> " << output_cloud->size() << " points" << std::endl;
}

void PowerlineCoarseExtractor::filterPointCloudByRange(const pcl::PointCloud<pcl::PointXYZI>::Ptr& input_cloud,
                                                     pcl::PointCloud<pcl::PointXYZI>::Ptr& output_cloud) {
    output_cloud->clear();
    output_cloud->reserve(input_cloud->size()); // 预分配内存提高性能
    
    for (const auto& point : input_cloud->points) {
        // 计算距离原点的距离
        double range = std::sqrt(point.x * point.x + point.y * point.y + point.z * point.z);
        
        // 检查距离和高度范围
        if (range >= min_range_ && range <= max_range_ && 
            point.z >= min_height_ && point.z <= max_height_) {
            output_cloud->push_back(point);
        }
    }
    
    output_cloud->width = output_cloud->size();
    output_cloud->height = 1;
    output_cloud->is_dense = true;
}

void PowerlineCoarseExtractor::downsamplePointCloud(const pcl::PointCloud<pcl::PointXYZI>::Ptr& input_cloud,
                                                  pcl::PointCloud<pcl::PointXYZI>::Ptr& output_cloud) {
    if (input_cloud->empty()) {
        output_cloud->clear();
        return;
    }
    
    pcl::VoxelGrid<pcl::PointXYZI> voxel_filter;
    voxel_filter.setInputCloud(input_cloud);
    voxel_filter.setLeafSize(voxel_size_, voxel_size_, voxel_size_);
    voxel_filter.filter(*output_cloud);
}

void PowerlineCoarseExtractor::removeOutliers(const pcl::PointCloud<pcl::PointXYZI>::Ptr& input_cloud,
                                            pcl::PointCloud<pcl::PointXYZI>::Ptr& output_cloud) {
    if (input_cloud->empty()) {
        output_cloud->clear();
        return;
    }
    
    // 如果点云太小，跳过离群点移除
    if (input_cloud->size() < static_cast<size_t>(outlier_mean_k_)) {
        *output_cloud = *input_cloud;
        return;
    }
    
    pcl::StatisticalOutlierRemoval<pcl::PointXYZI> sor;
    sor.setInputCloud(input_cloud);
    sor.setMeanK(outlier_mean_k_);
    sor.setStddevMulThresh(outlier_std_thresh_);
    sor.filter(*output_cloud);
}

void PowerlineCoarseExtractor::extractByLinearity(const pcl::PointCloud<pcl::PointXYZI>::Ptr& input_cloud,
                                                pcl::PointCloud<pcl::PointXYZI>::Ptr& output_cloud) {
    output_cloud->clear();
    
    if (input_cloud->empty()) {
        return;
    }
    
    // 设置KD树
    kdtree_->setInputCloud(input_cloud);
    
    std::vector<int> point_idx_radius_search;
    std::vector<float> point_radius_squared_distance;
    
    // 预分配内存
    output_cloud->reserve(input_cloud->size() / 10); // 估计10%的点是电力线
    
    // 对每个点计算局部线性度
    for (size_t i = 0; i < input_cloud->size(); ++i) {
        const pcl::PointXYZI& current_point = input_cloud->points[i];
        
        // 搜索半径内的近邻点
        int neighbors_found = kdtree_->radiusSearch(current_point, pca_radius_, 
                                                   point_idx_radius_search, 
                                                   point_radius_squared_distance);
        
        // 需要至少3个点进行PCA分析
        if (neighbors_found < 3) {
            continue;
        }
        
        // 计算线性度
        double linearity = calculateLinearity(current_point, point_idx_radius_search, input_cloud);
        
        // 根据线性度阈值判断是否为电力线点
        if (linearity > linearity_threshold_) {
            pcl::PointXYZI powerline_point = current_point;
            // 将线性度值存储在intensity字段中，便于后续分析和可视化
            powerline_point.intensity = static_cast<float>(linearity);
            output_cloud->push_back(powerline_point);
        }
    }
    
    output_cloud->width = output_cloud->size();
    output_cloud->height = 1;
    output_cloud->is_dense = true;
    
    std::cout << "Linearity-based extraction: " << input_cloud->size() 
              << " -> " << output_cloud->size() << " powerline candidate points" << std::endl;
}

double PowerlineCoarseExtractor::calculateLinearity(const pcl::PointXYZI& center_point,
                                                  const std::vector<int>& neighbors,
                                                  const pcl::PointCloud<pcl::PointXYZI>::Ptr& cloud) {
    // 构建近邻点矩阵
    Eigen::MatrixXf neighborhood(neighbors.size(), 3);
    for (size_t i = 0; i < neighbors.size(); ++i) {
        const pcl::PointXYZI& point = cloud->points[neighbors[i]];
        neighborhood(i, 0) = point.x;
        neighborhood(i, 1) = point.y;
        neighborhood(i, 2) = point.z;
    }
    
    // 计算中心点（质心）
    Eigen::Vector3f centroid = neighborhood.colwise().mean();
    
    // 中心化数据
    Eigen::MatrixXf centered = neighborhood.rowwise() - centroid.transpose();
    
    // 计算协方差矩阵
    Eigen::Matrix3f covariance = (centered.transpose() * centered) / static_cast<float>(neighbors.size() - 1);
    
    // 特征值分解
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> eigen_solver(covariance);
    Eigen::Vector3f eigenvalues = eigen_solver.eigenvalues();
    
    // 确保特征值按降序排列
    std::sort(eigenvalues.data(), eigenvalues.data() + 3, std::greater<float>());
    
    // 计算线性度：λ1 - λ2 / λ1
    // λ1是最大特征值，λ2是第二大特征值
    float lambda1 = eigenvalues(0);
    float lambda2 = eigenvalues(1);
    float lambda3 = eigenvalues(2);
    
    // 避免除零错误
    if (lambda1 < 1e-8) {
        return 0.0;
    }
    
    // 线性度计算
    double linearity = static_cast<double>((lambda1 - lambda2) / lambda1);
    
    // 确保线性度在[0,1]范围内
    linearity = std::max(0.0, std::min(1.0, linearity));
    
    return linearity;
}