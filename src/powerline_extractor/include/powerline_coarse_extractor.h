#ifndef POWERLINE_COARSE_EXTRACTOR_H
#define POWERLINE_COARSE_EXTRACTOR_H

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/filters/passthrough.h>

#include <Eigen/Dense>
#include <vector>
#include <memory>

/**
 * @brief 电力线粗提取类
 * 
 * 基于PCA线性度分析进行电力线的粗提取，主要用于从激光雷达点云中
 * 快速识别出具有线性特征的点，作为后续精细提取的预处理步骤。
 */
class PowerlineCoarseExtractor {
public:
    /**
     * @brief 构造函数
     */
    PowerlineCoarseExtractor();
    
    /**
     * @brief 析构函数
     */
    ~PowerlineCoarseExtractor();
    
    /**
     * @brief 设置PCA搜索半径
     * @param radius PCA近邻搜索半径
     */
    void setPCARadius(double radius);
    
    /**
     * @brief 设置线性度阈值
     * @param threshold 线性度阈值，范围[0,1]，值越大要求线性度越高
     */
    void setLinearityThreshold(double threshold);
    
    /**
     * @brief 设置体素滤波器叶子大小
     * @param leaf_size 体素大小
     */
    void setVoxelSize(double leaf_size);
    
    /**
     * @brief 设置距离过滤参数
     * @param min_range 最小距离
     * @param max_range 最大距离
     */
    void setRangeFilter(double min_range, double max_range);
    
    /**
     * @brief 设置高度过滤参数
     * @param min_height 最小高度
     * @param max_height 最大高度
     */
    void setHeightFilter(double min_height, double max_height);
    
    /**
     * @brief 设置离群点移除参数
     * @param mean_k 近邻点数量
     * @param std_thresh 标准差阈值
     */
    void setOutlierRemovalParams(int mean_k, double std_thresh);
    
    /**
     * @brief 执行电力线粗提取
     * @param input_cloud 输入点云
     * @param output_cloud 输出的电力线候选点云
     * @return 提取是否成功
     */
    bool extractPowerlines(const pcl::PointCloud<pcl::PointXYZI>::Ptr& input_cloud,
                          pcl::PointCloud<pcl::PointXYZI>::Ptr& output_cloud);
    
    /**
     * @brief 获取预处理后的点云（用于调试）
     * @return 预处理后的点云
     */
    pcl::PointCloud<pcl::PointXYZI>::Ptr getPreprocessedCloud() const;
    
    /**
     * @brief 获取提取统计信息
     * @param input_size 输入点云大小
     * @param preprocessed_size 预处理后点云大小  
     * @param output_size 输出点云大小
     */
    void getExtractionStats(size_t& input_size, size_t& preprocessed_size, size_t& output_size) const;

private:
    /**
     * @brief 预处理点云
     * @param input_cloud 输入点云
     * @param output_cloud 预处理后的点云
     */
    void preprocessPointCloud(const pcl::PointCloud<pcl::PointXYZI>::Ptr& input_cloud,
                            pcl::PointCloud<pcl::PointXYZI>::Ptr& output_cloud);
    
    /**
     * @brief 距离和高度过滤
     * @param input_cloud 输入点云
     * @param output_cloud 过滤后的点云
     */
    void filterPointCloudByRange(const pcl::PointCloud<pcl::PointXYZI>::Ptr& input_cloud,
                               pcl::PointCloud<pcl::PointXYZI>::Ptr& output_cloud);
    
    /**
     * @brief 体素下采样
     * @param input_cloud 输入点云
     * @param output_cloud 下采样后的点云
     */
    void downsamplePointCloud(const pcl::PointCloud<pcl::PointXYZI>::Ptr& input_cloud,
                            pcl::PointCloud<pcl::PointXYZI>::Ptr& output_cloud);
    
    /**
     * @brief 移除离群点
     * @param input_cloud 输入点云
     * @param output_cloud 移除离群点后的点云
     */
    void removeOutliers(const pcl::PointCloud<pcl::PointXYZI>::Ptr& input_cloud,
                      pcl::PointCloud<pcl::PointXYZI>::Ptr& output_cloud);
    
    /**
     * @brief 基于PCA线性度进行电力线提取
     * @param input_cloud 输入点云
     * @param output_cloud 提取的电力线点云
     */
    void extractByLinearity(const pcl::PointCloud<pcl::PointXYZI>::Ptr& input_cloud,
                          pcl::PointCloud<pcl::PointXYZI>::Ptr& output_cloud);
    
    /**
     * @brief 计算点的局部线性度
     * @param center_point 中心点
     * @param neighbors 近邻点集合
     * @return 线性度值，范围[0,1]
     */
    double calculateLinearity(const pcl::PointXYZI& center_point,
                            const std::vector<int>& neighbors,
                            const pcl::PointCloud<pcl::PointXYZI>::Ptr& cloud);

private:
    // PCA参数
    double pca_radius_;           ///< PCA搜索半径
    double linearity_threshold_;  ///< 线性度阈值
    
    // 预处理参数
    double voxel_size_;          ///< 体素滤波器大小
    double min_range_;           ///< 最小距离
    double max_range_;           ///< 最大距离
    double min_height_;          ///< 最小高度
    double max_height_;          ///< 最大高度
    
    // 离群点移除参数
    int outlier_mean_k_;         ///< 离群点检测近邻数量
    double outlier_std_thresh_;  ///< 离群点检测标准差阈值
    
    // 内部存储
    pcl::PointCloud<pcl::PointXYZI>::Ptr preprocessed_cloud_; ///< 预处理后的点云
    
    // 统计信息
    size_t input_size_;          ///< 输入点云大小
    size_t preprocessed_size_;   ///< 预处理后点云大小
    size_t output_size_;         ///< 输出点云大小
    
    // PCL对象
    pcl::KdTreeFLANN<pcl::PointXYZI>::Ptr kdtree_; ///< KD树用于近邻搜索
};

#endif // POWERLINE_COARSE_EXTRACTOR_H