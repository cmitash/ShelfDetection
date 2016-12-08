#include "apc_object.hpp"
#include "apc_shelf.hpp"
#include <pcl/point_types_conversion.h>

apc_shelf *shelf;
apc_object* train_objects[NUMBER_OBJECTS];

PointCloud::Ptr extractFeatures(PointCloudRGB::Ptr cloud_xyzrgb, int bin, vector<ObjectLabels> objects, ObjectLabels target)
{	
	//Estimating the normals
	PointCloud::Ptr cloud_xyz(new PointCloud);

	pcl::copyPointCloud(*cloud_xyzrgb, *cloud_xyz);
	pcl::search::Search<pcl::PointXYZ>::Ptr tree = boost::shared_ptr<pcl::search::Search<pcl::PointXYZ> > (new pcl::search::KdTree<pcl::PointXYZ>);
    pcl::PointCloud <pcl::Normal>::Ptr normals (new pcl::PointCloud <pcl::Normal>);
    pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> normal_estimator;
    normal_estimator.setSearchMethod (tree);
    normal_estimator.setInputCloud (cloud_xyz);
    normal_estimator.setRadiusSearch(0.005);
    normal_estimator.compute (*normals);
    
    ofstream myfile;
  	myfile.open ("features.txt");

  	PointCloudRGB::Ptr targetob(new PointCloudRGB);
  	PointCloud::Ptr targetobxyz(new PointCloud);

	for(int pos = 0;pos<cloud_xyzrgb->points.size();pos++){
		Eigen::Vector3d feature;
		
		//Get the color feature
		cv::Mat rgb_point = Mat::zeros(1, 1, CV_32F);
		cv::Mat hsv_point = Mat::zeros(1, 1, CV_8UC1);

		cv::Mat red = Mat::zeros(1, 1, CV_8UC1);
		cv::Mat green = Mat::zeros(1, 1, CV_8UC1);
		cv::Mat blue = Mat::zeros(1, 1, CV_8UC1);
		//double blue, red, green;
        uint32_t rgb_val_;
        memcpy(&rgb_val_, &(cloud_xyzrgb->points[pos].rgb), sizeof(uint32_t));
        blue.at<uchar>(0,0) = rgb_val_ & 0x000000ff;
        rgb_val_ = rgb_val_ >> 8;
        green.at<uchar>(0,0) = rgb_val_ & 0x000000ff;
        rgb_val_ = rgb_val_ >> 8;
        red.at<uchar>(0,0) = rgb_val_ & 0x000000ff;

        vector<Mat> channels;
	    channels.push_back(blue);
	    channels.push_back(green);
	    channels.push_back(red);
	 
	    /// Merge the three channels
	    merge(channels, rgb_point);

		cv::cvtColor(rgb_point, hsv_point, cv::COLOR_BGR2HSV);
		std::vector<cv::Mat> hsv_channels;
		cv::split(hsv_point, hsv_channels);
		int hue = (int)hsv_channels[0].at<uchar>(0,0);
		int sat = (int)hsv_channels[1].at<uchar>(0,0);
		int val = (int)hsv_channels[2].at<uchar>(0,0);
		
		if( sat > 60 && val > 20 )
			feature(0) = hue;
		else if(val > 200)
			feature(0) = HUE_WHITE;
		else if(val > 50 && val < 200)
			feature(0) = HUE_GRAY;
		else if(val < 50)
			feature(0) = HUE_BLACK;

		//Get height from shelf
		// double shelf_height = shelf->getShelfBinHeight(bin);
		// Eigen::Vector3d point(cloud_xyzrgb->points[pos].x, cloud_xyzrgb->points[pos].y,
		// 							cloud_xyzrgb->points[pos].z);
		// applyTF(point, point, shelf->CameraToWorldTf);
		// double value = abs((point[2] - shelf_height)/0.005);
		// if(value > 60)value = 61;
		// feature(1) = value;

		//Get the Curvature from Normal
 		double value = (*normals)[pos].curvature;
 		if(value != value)value = 0;;//What to do with NaN ?
 		value = abs(value/0.01);
 		if(value > 20)value = 21;
 		feature(1) = value;

 		double norm = 0.0;
 		double targetprob = 0.0;

 		std::vector<double> posterior(objects.size(),0);
 		for(int i = 0;i<objects.size();i++)
 		{
 			int obindex = objects[i];
 			posterior[i] = (train_objects[obindex]->getColorProbability(feature(0)) *
 								train_objects[obindex]->getNormalProbability(feature(1)) /**
 								train_objects[obindex]->getNormalProbability(feature(2))*/ );
 			norm += posterior[i];
 			if(obindex == target)
 				targetprob = posterior[i];
 		}

 		if(norm != 0)
 		{
 			targetprob /= norm;
 			int flag = 0;
 			for(int i = 0;i<objects.size();i++)
 			{
 				posterior[i] /= norm;
 				if(posterior[i] > targetprob){
 					flag = 1;
 					break;
 				}
 			}
 			if(!flag){
 				targetob->points.push_back(cloud_xyzrgb->points[pos]);
 			}
 		}

 		myfile<<"Features : "<<feature(0)<<" Normal : "<<feature(1)<<" target prob : "<<targetprob<<std::endl;
	}
	std::cout<<"DONE !! "<<std::endl;

	pcl::copyPointCloud(*targetob, *targetobxyz);
	return targetobxyz;
}

void trainObject(ObjectLabels index, string path, tf::StampedTransform transform, int binsarray[], int train)
{
	if(train == 1)
	{
		std::vector<int> bins(binsarray, binsarray + sizeof(binsarray) / sizeof(int) );
		train_objects[index] = new apc_object(index, path, transform, bins);
		train_objects[index]->LoadData();
		train_objects[index]->getColorFeatures();
		//train_objects[index]->getShelfHeightFeature(shelf);
		train_objects[index]->getNormalCurvature();
	}
	else
	{
		std::vector<int> bins(binsarray, binsarray + sizeof(binsarray) / sizeof(int) );
		train_objects[index] = new apc_object(index, path, transform, bins);
		train_objects[index]->ReadHistograms();
	}
}
 
int main(int argc, char **argv)
{
	std::string node_name = "Detection";

	ros::init(argc, argv, "ObjectDetection");
	ros::NodeHandle node;

	//Set params with shelf yaml file
	
	//Create a shelf instance and get the pose
	shelf = new apc_shelf();
	get_transform("map", "kinect2_head_rgb_optical_frame", shelf->CameraToWorldTf);

    sensor_msgs::PointCloud2::ConstPtr msg = 
    ros::topic::waitForMessage<sensor_msgs::PointCloud2>("/kinect2_head/hd/points", node, ros::Duration(6.0));

    PointCloudRGB::Ptr cloud_xyzrgb(new PointCloudRGB);
    PointCloudRGB::Ptr cloud_backup(new PointCloudRGB);
    PointCloud::Ptr cloud_xyz(new PointCloud);
    pcl::PCLPointCloud2 pcl_pc2;
    pcl_conversions::toPCL(*msg,pcl_pc2);
    pcl::fromPCLPointCloud2(pcl_pc2,*cloud_xyzrgb);
    pcl::copyPointCloud(*cloud_xyzrgb, *cloud_xyz);
    pcl::copyPointCloud(*cloud_xyzrgb, *cloud_backup);
    shelf->GetLeftRightPoses(cloud_xyz);
    geometry_msgs::PoseStamped shelfPose;
    shelf->smoothenShelfEstimate(shelfPose);

    int train = 0;
    //Train Object Models - 1
	std::string path = "/home/chaitanya/objectdetect/object-shelf-detection/src/training/data";
	int binsarray1[] = {1,1,1,1};
	trainObject(PRX_PAPER_TOWEL, path, shelf->CameraToWorldTf, binsarray1, train);
	
	//Train Object Models - 2
	int binsarray2[] = {2,2,2,2};
	trainObject(PRX_CLEAN_BRUSH, path, shelf->CameraToWorldTf, binsarray2, train);

	//Train Object Models - 3
	int binsarray3[] = {3,3,3};
	trainObject(PRX_DUMBBELLS, path, shelf->CameraToWorldTf, binsarray3, train);

	//Train Object Models - 4
	int binsarray4[] = {1,1,1};
	trainObject(PRX_BUBBLE_MAILER, path, shelf->CameraToWorldTf, binsarray4, train);

	//Train Object Models - 5
	int binsarray5[] = {1,1,1};
	trainObject(PRX_COMMAND_HANG, path, shelf->CameraToWorldTf, binsarray5, train);

	//Train Object Models - 6
	int binsarray6[] = {2,2,2,2};
	trainObject(PRX_STEMS, path, shelf->CameraToWorldTf, binsarray6, train);

	//Train Object Models - 7
	int binsarray7[] = {2,2,2,2,2};
	trainObject(PRX_CURTAIN_LINER, path, shelf->CameraToWorldTf, binsarray7, train);

	//Train Object Models - 8
	int binsarray8[] = {1,1,1,1};
	trainObject(PRX_SOCKS, path, shelf->CameraToWorldTf, binsarray8, train);

	//Train Object Models - 9
	int binsarray9[] = {3,3,3,3};
	trainObject(PRX_PLUG_PROTECTOR, path, shelf->CameraToWorldTf, binsarray9, train);

	//Train Object Models - 10
	int binsarray10[] = {1,1,1,1};
	trainObject(PRX_NONE, path, shelf->CameraToWorldTf, binsarray10, train);

	//Test Object Detection
    shelf->getShelfBinPoints(cloud_xyzrgb,1);
    vector<ObjectLabels> object_list;
    object_list.push_back(PRX_BUBBLE_MAILER);
    object_list.push_back(PRX_COMMAND_HANG);
    object_list.push_back(PRX_NONE);
    PointCloud::Ptr extract = extractFeatures(cloud_xyzrgb, 1, object_list, PRX_COMMAND_HANG);

    view2clouds(cloud_backup, extract);

   //  boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer;
   //  viewer = rgbVis(cloud_xyzrgb);

   //  while (!viewer->wasStopped ())
  	// {
   //  	viewer->spinOnce (100);
   //   	boost::this_thread::sleep (boost::posix_time::microseconds (100000));
  	// }

	ros::spin();
    ros::shutdown();
	return 0;
}