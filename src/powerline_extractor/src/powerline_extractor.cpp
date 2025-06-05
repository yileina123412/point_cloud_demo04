#include "powerline_extractor.h"
#include "powerline_coarse_extractor.h"

PowerlineExtractor::PowerlineExtractor(ros::NodeHandle& nh, ros::NodeHandle& private_nh)
    : nh_(nh), 
      private_nh_(private_nh),
      first_cloud_received_(false),
      obstacle_cluster_output_cloud(new pcl::PointCloud<pcl::PointXYZI>),
      static_map(new pcl::PointCloud<pcl::PointXYZI>),
      dynamic_map(new pcl::PointCloud<pcl::PointXYZI>),
      original_cloud_(new pcl::PointCloud<pcl::PointXYZI>()),
      preprocessed_cloud_(new pcl::PointCloud<pcl::PointXYZI>()),
      non_ground_cloud_(new pcl::PointCloud<pcl::PointXYZI>()),
      powerline_cloud_(new pcl::PointCloud<pcl::PointXYZI>()),
      clustered_powerline_cloud_(new pcl::PointCloud<pcl::PointXYZI>()),
      fine_extract_cloud_(new pcl::PointCloud<pcl::PointXYZI>()),
      filtered_pc_(new pcl::PointCloud<pcl::PointXYZI>()) {
    
    // 初始化TF
    tf_buffer_ = std::make_unique<tf2_ros::Buffer>();
    tf_listener_ = std::make_unique<tf2_ros::TransformListener>(*tf_buffer_);
    
    // 加载参数
    loadParameters();
    //初始化累积点云
    initializeAccumulateCloud();
    
    // 初始化粗提取器
    initializeCoarseExtractor();
    // 初始化精提取器
    initializeFineExtractor();
    
    // 初始化发布器和订阅器
    initializePublishers();
    initializeSubscribers();
    
    ROS_INFO("PowerlineExtractor initialized successfully");
}

PowerlineExtractor::~PowerlineExtractor() {
    ROS_INFO("PowerlineExtractor destructor called");
}

void PowerlineExtractor::loadParameters() {
    // 雷达相关参数
    private_nh_.param<std::string>("lidar_topic", lidar_topic_, "/livox/lidar");
    private_nh_.param<std::string>("lidar_frame", lidar_frame_, "livox_frame");
    private_nh_.param<std::string>("target_frame", target_frame_, "map");
    
    // 预处理参数
    private_nh_.param<double>("voxel_size", voxel_size_, 0.05);
    private_nh_.param<double>("min_range", min_range_, 0.5);
    private_nh_.param<double>("max_range", max_range_, 100.0);
    private_nh_.param<double>("min_height", min_height_, -5.0);
    private_nh_.param<double>("max_height", max_height_, 50.0);
    
    // PCA参数（用于粗提取）
    private_nh_.param<double>("pca_radius", pca_radius_, 0.5);
    private_nh_.param<double>("linearity_threshold", linearity_threshold_, 0.7);
    
    // 聚类参数
    private_nh_.param<double>("cluster_tolerance", cluster_tolerance_, 2.0);
    private_nh_.param<int>("min_cluster_size", min_cluster_size_, 15);
    private_nh_.param<int>("max_cluster_size", max_cluster_size_, 100000);
    
    // 离群点移除参数
    private_nh_.param<int>("outlier_mean_k", outlier_mean_k_, 50);
    private_nh_.param<double>("outlier_std_thresh", outlier_std_thresh_, 1.0);
    
    // 处理频率
    private_nh_.param<double>("process_frequency", process_frequency_, 2.0);
    
    // 打印参数
    ROS_INFO("=== Powerline Extractor Parameters ===");
    ROS_INFO("Lidar topic: %s", lidar_topic_.c_str());
    ROS_INFO("Lidar frame: %s", lidar_frame_.c_str());
    ROS_INFO("Target frame: %s", target_frame_.c_str());
    ROS_INFO("Voxel size: %.3f", voxel_size_);
    ROS_INFO("Range filter: [%.1f, %.1f]", min_range_, max_range_);
    ROS_INFO("Height filter: [%.1f, %.1f]", min_height_, max_height_);
    ROS_INFO("PCA radius: %.3f", pca_radius_);
    ROS_INFO("Linearity threshold: %.3f", linearity_threshold_);
    ROS_INFO("Cluster tolerance: %.3f", cluster_tolerance_);
    ROS_INFO("Cluster size range: [%d, %d]", min_cluster_size_, max_cluster_size_);
    ROS_INFO("Process frequency: %.1f Hz", process_frequency_);
}

void PowerlineExtractor::initializeCoarseExtractor() {
    // 创建粗提取器
    coarse_extractor_ = std::make_unique<PowerlineCoarseExtractor>();
    
    // 配置粗提取器参数
    coarse_extractor_->setPCARadius(pca_radius_);
    coarse_extractor_->setLinearityThreshold(linearity_threshold_);
    coarse_extractor_->setVoxelSize(voxel_size_);
    coarse_extractor_->setRangeFilter(min_range_, max_range_);
    coarse_extractor_->setHeightFilter(min_height_, max_height_);
    coarse_extractor_->setOutlierRemovalParams(outlier_mean_k_, outlier_std_thresh_);
    
    ROS_INFO("Coarse extractor initialized with parameters");
}
void PowerlineExtractor::initializeCoarseFilter()
{
    coarse_filter_ = std::make_unique<PowerLineFilter>(nh_);
}

void PowerlineExtractor::initializeFineExtractor(){

    fine_extractor_ = std::make_unique<PowerLineFineExtractor>(nh_);

    ROS_INFO("Fine extractor initialized successfully");


}

void PowerlineExtractor::initializePublishers() {

    obstacle_cluster_cloud_pub_ = private_nh_.advertise<sensor_msgs::PointCloud2>("obstacle_cluster_cloud", 1);
    static_cloud_pub_ = private_nh_.advertise<sensor_msgs::PointCloud2>("static_cloud", 1);
    dynamic_cloud_pub_ = private_nh_.advertise<sensor_msgs::PointCloud2>("dynamic_cloud", 1);

    original_cloud_pub_ = private_nh_.advertise<sensor_msgs::PointCloud2>("original_cloud", 1);
    preprocessed_cloud_pub_ = private_nh_.advertise<sensor_msgs::PointCloud2>("preprocessed_cloud", 1);
    powerline_cloud_pub_ = private_nh_.advertise<sensor_msgs::PointCloud2>("powerline_cloud", 1);
    clustered_powerline_cloud_pub_ = private_nh_.advertise<sensor_msgs::PointCloud2>("clustered_powerline_cloud", 1);

    //pca粗提取后过滤
    coarse_filter_cloud_pub_ = private_nh_.advertise<sensor_msgs::PointCloud2>("coarse_filter_cloud", 1);
    // 累积点云的发布器
    accumulated_cloud_pub_ = private_nh_.advertise<sensor_msgs::PointCloud2>("accumulated_cloud", 1);
    //octree累积点云的发布器
    octree_accumulated_cloud_pub_ = private_nh_.advertise<sensor_msgs::PointCloud2>("octree_accumulated_cloud", 1);
    fine_extractor_cloud_pub_ = private_nh_.advertise<sensor_msgs::PointCloud2>("fine_extractor_cloud", 1);
    ROS_INFO("Publishers initialized");
}
void PowerlineExtractor::initializeAccumulateCloud()
{

    // 初始化障碍物过滤器
    obstacle_cluster_.reset(new ObstacleClustering(nh_));
    // octree_accumulator_.reset(new powerline_extractor::PointCloudAccumulatorOctree(nh_));

    // ROS_INFO("Octree Accumulate Cloud initialized");

    // accumulator_.reset(new PointCloudAccumulator());
    // // 从参数服务器加载参数
    // accumulator_->loadParamsFromServer(nh_);

    map_builder_.reset(new MapBuilder());

    ROS_INFO("Accumulate Cloud initialized");


}
void PowerlineExtractor::initializeSubscribers() {
    point_cloud_sub_ = nh_.subscribe(lidar_topic_, 1, &PowerlineExtractor::pointCloudCallback, this);
    ROS_INFO("Subscribed to point cloud topic: %s", lidar_topic_.c_str());
}

void PowerlineExtractor::pointCloudCallback(const sensor_msgs::PointCloud2::ConstPtr& msg) {
    // 频率控制
    ros::Time current_time = ros::Time::now();
    if (first_cloud_received_ && 
        (current_time - last_process_time_).toSec() < (1.0 / process_frequency_)) {
        return;
    }
    
    last_process_time_ = current_time;
    first_cloud_received_ = true;
    
    ROS_DEBUG("Received point cloud with %d points", msg->width * msg->height);
    
    try {
        // 坐标变换
        sensor_msgs::PointCloud2 transformed_msg;
        if (!transformPointCloud(msg, transformed_msg)) {
            ROS_WARN("Failed to transform point cloud, skipping this frame");
            return;
        }
        
        // 转换为PCL格式
        pcl::fromROSMsg(transformed_msg, *original_cloud_);
        
        if (original_cloud_->empty()) {
            ROS_WARN("Received empty point cloud");
            return;
        }


        obstacle_cluster_->process(original_cloud_, obstacle_cluster_output_cloud, excluded_regions);

        map_builder_->processPointCloud(obstacle_cluster_output_cloud,static_map,dynamic_map);

        // accumulator_->addPointCloud(original_cloud_, msg->header.stamp);
        // // 获取累积的稳定点云
        // pcl::PointCloud<pcl::PointXYZI>::Ptr stable_cloud = accumulator_->getStableCloud();

        // if (!stable_cloud || stable_cloud->empty()) {
        //     ROS_WARN("Accumulate cloud returned empty cloud, skipping this frame");
        //     return;
        // }

        // octree_accumulator_->addPointCloud(original_cloud_,msg->header.stamp);

        // pcl::PointCloud<pcl::PointXYZI>::Ptr octree_accumulated_cloud_temp = octree_accumulator_->getAccumulatedCloud();
        // if(octree_accumulated_cloud_temp->empty()){
        //     ROS_WARN("Octree Accumulate cloud returned empty cloud, skipping this frame");
        //     return;
        // }

        
        // 使用粗提取器进行电力线粗提取
        // if (!coarse_extractor_->extractPowerlines(original_cloud_, powerline_cloud_)) {
        //     ROS_WARN("Coarse extraction failed, skipping this frame");
        //     return;
        // }
    

        if (!coarse_extractor_->extractPowerlines(static_map, powerline_cloud_)) {
            ROS_WARN("Coarse extraction failed, skipping this frame");
            return;
        }
        
        // 获取预处理后的点云（用于可视化）
        preprocessed_cloud_ = coarse_extractor_->getPreprocessedCloud();
        
        // 聚类处理
        clusterPowerlines(powerline_cloud_, clustered_powerline_cloud_);
        
        //对粗提取的pca再次过滤
        // coarse_filter_->filter(clustered_powerline_cloud_,static_map,filtered_pc_);

        // 发布结果
        publishPointClouds(original_cloud_, preprocessed_cloud_, powerline_cloud_, 
                          clustered_powerline_cloud_, transformed_msg.header);
        
        // 精提取
        fine_extractor_->extractPowerLines(clustered_powerline_cloud_,fine_extract_cloud_);
        
        
        // 获取并打印统计信息
        size_t input_size, preprocessed_size, output_size;
        coarse_extractor_->getExtractionStats(input_size, preprocessed_size, output_size);
        
        ROS_DEBUG("Processed point cloud: original(%zu), preprocessed(%zu), powerline(%zu), clustered(%zu)",
                 input_size, preprocessed_size, output_size, clustered_powerline_cloud_->size());
                 
    } catch (const std::exception& e) {
        ROS_ERROR("Error processing point cloud: %s", e.what());
    }
}

bool PowerlineExtractor::transformPointCloud(const sensor_msgs::PointCloud2::ConstPtr& input_msg,
                                            sensor_msgs::PointCloud2& transformed_msg) {
    try {
        // 检查是否需要变换
        if (input_msg->header.frame_id == target_frame_) {
            transformed_msg = *input_msg;
            return true;
        }
        
        // 查找变换
        geometry_msgs::TransformStamped transform_stamped;
        transform_stamped = tf_buffer_->lookupTransform(
            target_frame_, input_msg->header.frame_id, 
            input_msg->header.stamp, ros::Duration(1.0));
        
        // 执行变换
        tf2::doTransform(*input_msg, transformed_msg, transform_stamped);
        transformed_msg.header.frame_id = target_frame_;
        
        return true;
        
    } catch (tf2::TransformException& ex) {
        ROS_WARN("Could not transform point cloud: %s", ex.what());
        return false;
    }
}

void PowerlineExtractor::clusterPowerlines(const pcl::PointCloud<pcl::PointXYZI>::Ptr& input_cloud,
                                         pcl::PointCloud<pcl::PointXYZI>::Ptr& clustered_cloud) {
    clustered_cloud->clear();
    
    if (input_cloud->empty()) {
        return;
    }
    
    // 创建KD树用于欧几里得聚类
    pcl::search::KdTree<pcl::PointXYZI>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZI>);
    tree->setInputCloud(input_cloud);
    
    // 存储聚类结果
    std::vector<pcl::PointIndices> cluster_indices;
    
    // 创建欧几里得聚类对象
    pcl::EuclideanClusterExtraction<pcl::PointXYZI> ec;
    ec.setClusterTolerance(cluster_tolerance_);
    ec.setMinClusterSize(min_cluster_size_);
    ec.setMaxClusterSize(max_cluster_size_);
    ec.setSearchMethod(tree);
    ec.setInputCloud(input_cloud);
    ec.extract(cluster_indices);
    
    ROS_DEBUG("Found %zu clusters in powerline points", cluster_indices.size());
    
    // 处理每个聚类
    int cluster_id = 0;
    for (const auto& indices : cluster_indices) {
        cluster_id++;
        
        // 提取当前聚类的点
        for (const auto& index : indices.indices) {
            pcl::PointXYZI point = input_cloud->points[index];
            // 设置intensity值为聚类ID，便于可视化不同的聚类
            point.intensity = static_cast<float>(cluster_id);
            clustered_cloud->push_back(point);
        }
    }
    
    clustered_cloud->width = clustered_cloud->size();
    clustered_cloud->height = 1;
    clustered_cloud->is_dense = true;
    
    ROS_DEBUG("Clustering: %zu points -> %zu clusters -> %zu points",
             input_cloud->size(), cluster_indices.size(), clustered_cloud->size());
}

void PowerlineExtractor::publishPointClouds(const pcl::PointCloud<pcl::PointXYZI>::Ptr& original_cloud,
                                          const pcl::PointCloud<pcl::PointXYZI>::Ptr& preprocessed_cloud,
                                          const pcl::PointCloud<pcl::PointXYZI>::Ptr& powerline_cloud,
                                          const pcl::PointCloud<pcl::PointXYZI>::Ptr& clustered_cloud,
                                          const std_msgs::Header& header) {
    
    // 发布原始点云
    if (original_cloud_pub_.getNumSubscribers() > 0 && !original_cloud->empty()) {
        sensor_msgs::PointCloud2 original_msg;
        pcl::toROSMsg(*original_cloud, original_msg);
        original_msg.header = header;
        original_cloud_pub_.publish(original_msg);
    }
    if(obstacle_cluster_cloud_pub_.getNumSubscribers() > 0 && !obstacle_cluster_output_cloud->empty()){
        sensor_msgs::PointCloud2 obstacle_cluster_msg;
        pcl::toROSMsg(*obstacle_cluster_output_cloud, obstacle_cluster_msg);
        obstacle_cluster_msg.header = header;
        obstacle_cluster_cloud_pub_.publish(obstacle_cluster_msg);

    }



    if (static_cloud_pub_.getNumSubscribers() > 0 && !static_map->empty()) {
        sensor_msgs::PointCloud2 static_msg;
        pcl::toROSMsg(*static_map, static_msg);
        static_msg.header = header;
        static_cloud_pub_.publish(static_msg);
    }

    if (dynamic_cloud_pub_.getNumSubscribers() > 0 && !dynamic_map->empty()) {
        sensor_msgs::PointCloud2 dynamic_msg;
        pcl::toROSMsg(*dynamic_map, dynamic_msg);
        dynamic_msg.header = header;
        dynamic_cloud_pub_.publish(dynamic_msg);
    }

    if(coarse_filter_cloud_pub_.getNumSubscribers() > 0 && !filtered_pc_->empty()){

        sensor_msgs::PointCloud2 filtered_msg;
        pcl::toROSMsg(*filtered_pc_, filtered_msg);
        filtered_msg.header = header;
        coarse_filter_cloud_pub_.publish(filtered_msg);

    }
    

    // 发布预处理后的点云
    if (preprocessed_cloud_pub_.getNumSubscribers() > 0 && !preprocessed_cloud->empty()) {
        sensor_msgs::PointCloud2 preprocessed_msg;
        pcl::toROSMsg(*preprocessed_cloud, preprocessed_msg);
        preprocessed_msg.header = header;
        preprocessed_cloud_pub_.publish(preprocessed_msg);
    }
    
    // 发布电力线点云
    if (powerline_cloud_pub_.getNumSubscribers() > 0 && !powerline_cloud->empty()) {
        sensor_msgs::PointCloud2 powerline_msg;
        pcl::toROSMsg(*powerline_cloud, powerline_msg);
        powerline_msg.header = header;
        powerline_cloud_pub_.publish(powerline_msg);
    }
    
    // 发布聚类后的电力线点云
    if (clustered_powerline_cloud_pub_.getNumSubscribers() > 0 && !clustered_cloud->empty()) {
        sensor_msgs::PointCloud2 clustered_msg;
        pcl::toROSMsg(*clustered_cloud, clustered_msg);
        clustered_msg.header = header;
        clustered_powerline_cloud_pub_.publish(clustered_msg);
    }

    if(fine_extractor_cloud_pub_.getNumSubscribers()>0 && !fine_extract_cloud_->empty()){
        sensor_msgs::PointCloud2 fine_extractor_msg;
        pcl::toROSMsg(*fine_extract_cloud_, fine_extractor_msg);
        fine_extractor_msg.header = header;
        fine_extractor_cloud_pub_.publish(fine_extractor_msg);
    }
    
    // 打印统计信息
    static int frame_count = 0;
    frame_count++;
    if (frame_count % 10 == 0) {  // 每10帧打印一次
        ROS_INFO("Frame %d - Points: original(%zu), preprocessed(%zu), powerline(%zu), clustered(%zu)",
                 frame_count, original_cloud->size(), preprocessed_cloud->size(), 
                 powerline_cloud->size(), clustered_cloud->size());
    }
}