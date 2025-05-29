#ifndef ROUGH_EXTRACTION_H
#define ROUGH_EXTRACTION_H

#include <ros/ros.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <Eigen/Dense>

class RoughExtractor {
public:
    RoughExtractor(ros::NodeHandle& nh, ros::NodeHandle& private_nh);
    
    void extractPowerlinesByLinearity(
        const pcl::PointCloud<pcl::PointXYZI>::Ptr& input_cloud,
        pcl::PointCloud<pcl::PointXYZI>::Ptr& powerline_cloud);

private:
    void loadParameters();

    ros::NodeHandle nh_;
    ros::NodeHandle private_nh_;
    double linearity_threshold_;
};

#endif // ROUGH_EXTRACTION_H
