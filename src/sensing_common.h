#ifndef SENSING_COMMON
#define SENSING_COMMON

#include <iostream>
#include <fstream>
#include <vector>
#include <string>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <boost/thread/thread.hpp>
#include <pcl/common/common_headers.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/io/pcd_io.h>
//#include <pcl_ros/io/bag_io.h>
#include <pcl/conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/common/common.h>
#include <pcl_ros/transforms.h>
//#include <pcl/point_types_conversion.h>

#include <pcl/filters/passthrough.h>
#include <pcl/search/search.h>
#include <pcl/search/kdtree.h>
#include <pcl/features/normal_3d.h>
#include <pcl/segmentation/region_growing.h>
#include <pcl/filters/crop_box.h>

#include <pcl_ros/point_cloud.h>
#include <geometry_msgs/PoseStamped.h>

#include <pcl/segmentation/sac_segmentation.h>

#include <ros/ros.h>

#include <dirent.h>
#include <sstream>
#include <utility>

typedef pcl::PointCloud<pcl::PointXYZ> PointCloud;
typedef pcl::PointCloud<pcl::PointXYZRGB> PointCloudRGB;

#define PI 3.14
#define SHELF_NUM_SAMPLES 100
#define NUMBER_OBJECTS 20

void applyTF(PointCloudRGB::Ptr cloud_in, tf::Matrix3x3 rotation, Eigen::Vector3d origin);
void applyTF(PointCloud::Ptr cloud_in, tf::Matrix3x3 rotation, Eigen::Vector3d origin);
void applyTF(geometry_msgs::PoseStamped SrcPose, geometry_msgs::PoseStamped &DstPose, tf::Transform transform);
void applyTF(Eigen::Vector3d SrcPoint, Eigen::Vector3d& DstPoint, tf::Transform transform);
void PrintPose(geometry_msgs::PoseStamped pPose, std::string str);
bool get_transform(std::string Targetframe, std::string Srcframe, tf::StampedTransform& transform);
void viewShelfSegmentation(PointCloud::Ptr &cloud1, 
              PointCloudRGB::Ptr &cloud2,
              Eigen::Vector3d ltop,
              Eigen::Vector3d rtop,
              Eigen::Vector3d Normal,
              Eigen::Vector3d Edge,
              Eigen::Vector3d Cross);
void view2clouds(PointCloudRGB::Ptr &cloud1, 
              PointCloud::Ptr &cloud2);
boost::shared_ptr<pcl::visualization::PCLVisualizer> rgbVis 
				(pcl::PointCloud<pcl::PointXYZRGB>::ConstPtr cloud);
boost::shared_ptr<pcl::visualization::PCLVisualizer> normalsVis 
				(pcl::PointCloud<pcl::PointXYZ>::ConstPtr cloud, 
					pcl::PointCloud<pcl::Normal>::ConstPtr normals);
boost::shared_ptr<pcl::visualization::PCLVisualizer> simpleVis 
				(pcl::PointCloud<pcl::PointXYZ>::ConstPtr cloud);

void writeHistogramtoFile(std::string filename, cv::Mat Hist);
void readHistogramtoFile(std::string filename, cv::Mat &Hist);
#endif