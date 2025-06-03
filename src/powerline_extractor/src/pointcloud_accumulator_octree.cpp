#include "pointcloud_accumulator_octree.h"

namespace powerline_extractor {

PointCloudAccumulatorOctree::PointCloudAccumulatorOctree(ros::NodeHandle& nh)
    : nh_(nh), octree_(0.1f) // default resolution
{
    nh_.param("octree/accumulation_time", accumulation_time_, 0.8);
    nh_.param("octree/voxel_resolution", voxel_resolution_, 0.1);

    ROS_INFO("Octree PointCloud Accumulator parameters loaded:");
    ROS_INFO("  Accumulation time: %.1fs", accumulation_time_);
    ROS_INFO("  Voxel resolution: %.2fm", voxel_resolution_);

    octree_.setResolution(voxel_resolution_);
    accumulated_cloud_.reset(new pcl::PointCloud<pcl::PointXYZI>());
}

void PointCloudAccumulatorOctree::addPointCloud(const pcl::PointCloud<pcl::PointXYZI>::Ptr& cloud, const ros::Time& timestamp) {
    cloud_buffer_.emplace_back(timestamp, cloud);
    removeOldPointClouds(timestamp);
    updateOctree();
}

void PointCloudAccumulatorOctree::removeOldPointClouds(const ros::Time& current_time) {
    while (!cloud_buffer_.empty() && (current_time - cloud_buffer_.front().first).toSec() > accumulation_time_) {
        cloud_buffer_.pop_front();
    }
}

void PointCloudAccumulatorOctree::updateOctree() {
    octree_.deleteTree();
    accumulated_cloud_->clear();

    for (const auto& pair : cloud_buffer_) {
        *accumulated_cloud_ += *(pair.second);
    }

    octree_.setInputCloud(accumulated_cloud_);
    octree_.addPointsFromInputCloud();
}

pcl::PointCloud<pcl::PointXYZI>::Ptr PointCloudAccumulatorOctree::getAccumulatedCloud() {
    return accumulated_cloud_;
}

} // namespace powerline_extractor
