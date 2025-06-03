#ifndef POWERLINE_EXTRACTOR_POINTCLOUD_ACCUMULATOR_OCTREE_H
#define POWERLINE_EXTRACTOR_POINTCLOUD_ACCUMULATOR_OCTREE_H

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/octree/octree_pointcloud.h>
#include <deque>
#include <ros/ros.h>

namespace powerline_extractor {

class PointCloudAccumulatorOctree {
public:
    PointCloudAccumulatorOctree(ros::NodeHandle& nh);

    void addPointCloud(const pcl::PointCloud<pcl::PointXYZI>::Ptr& cloud, const ros::Time& timestamp);
    pcl::PointCloud<pcl::PointXYZI>::Ptr getAccumulatedCloud();

private:
    void removeOldPointClouds(const ros::Time& current_time);
    void updateOctree();

    ros::NodeHandle nh_;
    double accumulation_time_; // seconds
    double voxel_resolution_;  // meters

    std::deque<std::pair<ros::Time, pcl::PointCloud<pcl::PointXYZI>::Ptr>> cloud_buffer_;
    pcl::octree::OctreePointCloud<pcl::PointXYZI> octree_;
    pcl::PointCloud<pcl::PointXYZI>::Ptr accumulated_cloud_;
};

} // namespace powerline_extractor

#endif // POWERLINE_EXTRACTOR_POINTCLOUD_ACCUMULATOR_OCTREE_H
