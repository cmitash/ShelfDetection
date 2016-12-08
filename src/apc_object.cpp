#include "apc_object.hpp"

namespace patch
{
    template < typename T > std::string to_string( const T& n )
    {
        std::ostringstream stm ;
        stm << n ;
        return stm.str() ;
    }
}

apc_object::apc_object()
{
	label = PRX_NONE;
	TrainingMask.clear();
	TrainingImgNames.clear();
	bin.clear();
	ColorHist = Mat::zeros(183, 1, CV_32F);
	HeightHist = Mat::zeros(63, 1, CV_32F);
}

apc_object::apc_object(ObjectLabels Initlabel, std::string trainpath, tf::StampedTransform transform, std::vector<int> bins)
{
	label = Initlabel;
	TrainingMask.clear();
	TrainingImgNames.clear();
	bin.assign(bins.begin(), bins.end());
	ColorHist = Mat::zeros(183, 1, CV_32F);
	HeightHist = Mat::zeros(63, 1, CV_32F);
	NormalHist = Mat::zeros(21, 1, CV_32F);
	NaNHist = Mat::zeros(2, 1, CV_32F);
	path.assign(trainpath);
	CameraToWorldTf = transform;
}

apc_object::~apc_object() { }

void apc_object::LoadData()
{
	std::cout<<"Loading Data from following files : "<<std::endl;
	DIR* pDIR;
	struct dirent *entry;

	std::string location = path + "/" + patch::to_string(label);
	std::cout<<location<<std::endl;
	if( pDIR=opendir(location.c_str()) ){
		while(entry = readdir(pDIR)){
			std::string line;
			if(entry->d_type == DT_REG && strcmp(entry->d_name,"bin.txt") != 0){
				std::cout<<entry->d_name<<std::endl;
				cv::Mat image = cv::imread(location + "/" + entry->d_name, CV_8UC1);
				TrainingMask.push_back(image.clone());
				TrainingImgNames.push_back(entry->d_name);
			}	
		}
		closedir(pDIR);
	}
}

void apc_object::ShowTrainingImages()
{
	const std::string winName = "image";

	for(int i=0;i<TrainingImgNames.size();i++){
		cv::Mat res;
		cv::namedWindow(winName, cv::WINDOW_AUTOSIZE);
		std::string location = path + "/images/" + TrainingImgNames[i];
		cv::Mat image = cv::imread(location, CV_LOAD_IMAGE_COLOR);
		image.copyTo(res, TrainingMask[i]);
		cv::imshow(winName, res);
		cv::waitKey(0);
		cv::destroyWindow(winName);
	}
}

Mat apc_object::ConvertImgtoHeightImg(Mat image, double bin_height)
{
	double fx_d = 1.0 / 1051.89;
	double fy_d = 1.0 / 1060.192;
	double cx_d = 962.20;
	double cy_d = 535.165;

	cv::Mat height_image = Mat::zeros(image.rows, image.cols, CV_8UC1);

	for(int i=0; i<image.rows; i++)
		for(int j=0; j<image.cols; j++){
			Eigen::Vector3d point;
			double depth = image.at<float>(i,j);
			point << ((j - cx_d) * depth * fx_d), 
						((i - cy_d) * depth * fy_d), depth;
			point = point/1000;
			applyTF(point, point, CameraToWorldTf);
			point[2] = point[2] - bin_height;
			double val = abs(point[2]/0.005);

			if(val > 60)val = 61;
			height_image.at<uchar>(i,j) = val;
		}

    return height_image;
}

void apc_object::getShelfHeightFeature(apc_shelf *shelf)
{
	std::cout<<"Training Shelf Height feature"<<std::endl;
	for(int idx=0;idx<TrainingImgNames.size();idx++){
		std::string dep = TrainingImgNames[idx];
		dep.replace(0,5,"depth");
		std::string location = path + "/images/" + dep;;
		cv::Mat image = cv::imread(location, CV_LOAD_IMAGE_ANYDEPTH | CV_LOAD_IMAGE_ANYCOLOR);
		image.convertTo(image, CV_32F);
		double height = shelf->getShelfBinHeight(bin[idx]);
		cv::Mat heightImg = ConvertImgtoHeightImg(image, height);
		cv::Mat hist = calcHistogram(heightImg, TrainingMask[idx], 0, 62);
		HeightHist = HeightHist + hist;
	}
	std::cout<<"Height Histogram "<<std::endl;
	cv::GaussianBlur(HeightHist, HeightHist, Size(0,0), 1);
	double n = norm(HeightHist,NORM_L1);
	HeightHist = HeightHist/n;
	std::cout<<HeightHist<<std::endl;
	std::cout<<"*******************************"<<std::endl;
	
	std::string filename = "HeightHist"+ patch::to_string(label)+".txt";
	writeHistogramtoFile(filename, HeightHist);
}

void apc_object::getNormalCurvature()
{
	std::cout<<"Training Normal curvature feature"<<std::endl;
	for(int idx=0;idx<TrainingImgNames.size();idx++){
		std::string dep = TrainingImgNames[idx];
		dep.replace(0,5,"depth");
		std::string location = path + "/images/" + dep;
		cv::Mat image = cv::imread(location, CV_LOAD_IMAGE_ANYDEPTH | CV_LOAD_IMAGE_ANYCOLOR);
		image.convertTo(image, CV_32F);
		cv::Mat NormalImg = ConvertImgtoNormalImg(image, TrainingMask[idx]);
		NormalHist = NormalHist + NormalImg;
	}

	std::cout<<"Normal Histogram"<<std::endl;
	cv::GaussianBlur(NormalHist, NormalHist, Size(0,0), 1);
	double n = norm(NormalHist,NORM_L1);
	NormalHist = NormalHist/n;
	std::cout<<NormalHist<<std::endl;
	std::cout<<"*******************************"<<std::endl;
	
	std::string filename = "NormalHist"+ patch::to_string(label)+".txt";
	writeHistogramtoFile(filename, NormalHist);

	std::cout<<"NaN Histogram"<<std::endl;
	n = norm(NaNHist,NORM_L1);
	NaNHist = NaNHist/n;
	
	std::cout<<NaNHist<<std::endl;
	std::cout<<"*******************************"<<std::endl;
}

Mat apc_object::ConvertImgtoNormalImg(Mat image, Mat mask)
{
	double fx_d = 1.0 / 1051.89;
	double fy_d = 1.0 / 1060.192;
	double cx_d = 962.20;
	double cy_d = 535.165;

	cv::Mat normal_image = Mat::zeros(21, 1, CV_32F);
	PointCloud::Ptr cloud (new PointCloud);
	for(int i=0; i<image.rows; i++)
		for(int j=0; j<image.cols; j++){
			double depth = image.at<float>(i,j)/1000;
			if(mask.at<uchar>(i,j) == 1){
				if(depth  != 0){
				Eigen::Vector3d point;
				point << ((j - cx_d) * depth * fx_d), 
						((i - cy_d) * depth * fy_d), depth;
				applyTF(point, point, CameraToWorldTf);
				pcl::PointXYZ pt(point[0], point[1], point[2]);
				cloud->points.push_back(pt);
				NaNHist.at<float>(1,0) = NaNHist.at<float>(1,0) + 1;
				}else{
					NaNHist.at<float>(0,0) = NaNHist.at<float>(0,0) + 1;
				}
			}
		}

	pcl::search::Search<pcl::PointXYZ>::Ptr tree = boost::shared_ptr<pcl::search::Search<pcl::PointXYZ> > (new pcl::search::KdTree<pcl::PointXYZ>);
    pcl::PointCloud <pcl::Normal>::Ptr normals (new pcl::PointCloud <pcl::Normal>);
    pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> normal_estimator;
    normal_estimator.setSearchMethod(tree);
    normal_estimator.setInputCloud(cloud);
    normal_estimator.setRadiusSearch(0.005);
    normal_estimator.compute(*normals);

    for(int i = 0;i<cloud->points.size();i++)
    {
    	if((*normals)[i].curvature != (*normals)[i].curvature)continue;
    	double val = abs((*normals)[i].curvature/0.01);
    	if(val > 20)val = 21;
    	normal_image.at<float>(val,0) = normal_image.at<float>(val,0) + 1;
    }

	// boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer;
 	// viewer = normalsVis(cloud, normals);

 	// while (!viewer->wasStopped ())
 	// {
 	// viewer->spinOnce (100);
 	// boost::this_thread::sleep (boost::posix_time::microseconds (100000));
 	// }


    return normal_image;	
}

void apc_object::getColorFeatures()
{
	std::cout<<"Training Object Color feature"<<std::endl;
	cv::Mat hist1 = Mat::zeros(180, 1, CV_32F);
	cv::Mat hist2 = Mat::zeros(3, 1, CV_32F);

	for(int idx=0;idx<TrainingImgNames.size();idx++){
		cv::Mat hsv_image;

		std::string location = path + "/images/" + TrainingImgNames[idx];
		cv::Mat image = cv::imread(location, CV_LOAD_IMAGE_COLOR);
		cv::Mat hwgb_image = ConvertImgtoHWGB(image);

		cv::Mat thist1 = calcHistogram(hwgb_image, TrainingMask[idx], 0, 179);
		//cv::Mat thist2 = calcHistogram(hwgb_image, TrainingMask[idx], 180, 182);

		hist1 = thist1 + hist1;
		//hist2 = thist2 + hist2;
	}
	cv::GaussianBlur(hist1, hist1, Size(0,0), 1);
	vconcat(hist1,hist2,ColorHist);
	double n = norm(ColorHist,NORM_L1);
	ColorHist = ColorHist/n;
	std::cout<<"Color Histogram : "<<std::endl;
	std::cout<<ColorHist<<std::endl;
	std::cout<<"*******************************"<<std::endl;

	std::string filename = "ColorHist"+ patch::to_string(label)+".txt";
	writeHistogramtoFile(filename, ColorHist);

}

Mat apc_object::ConvertImgtoHWGB(Mat image)
{
	cv::Mat hsv_image;
	
	cv::cvtColor(image, hsv_image, cv::COLOR_BGR2HSV);
	std::vector<cv::Mat> hsv_channels;
	cv::split(hsv_image, hsv_channels);
	cv::Mat hwgb_image = Mat::zeros(hsv_image.rows, hsv_image.cols, CV_8UC1);
	cv::Mat h_image = Mat::zeros(hsv_image.rows, hsv_image.cols, CV_8UC1);
	
	for(int i=0; i<hsv_image.rows; i++)
		for(int j=0; j<hsv_image.cols; j++){
			int hue = (int)hsv_channels[0].at<uchar>(i,j);
			int sat = (int)hsv_channels[1].at<uchar>(i,j);
			int val = (int)hsv_channels[2].at<uchar>(i,j);

			if( sat > 60 && val > 20 )
				hwgb_image.at<uchar>(i,j) = hue;
			else if(val > 200)
			{
				hwgb_image.at<uchar>(i,j) = HUE_WHITE;
				h_image.at<uchar>(i,j) = 255;
			}
			else if(val > 50 && val < 200)
				hwgb_image.at<uchar>(i,j) = HUE_GRAY;
			else if(val < 50)
				hwgb_image.at<uchar>(i,j) = HUE_BLACK;
		}

	// DEBUG:: Display the component image
	// const std::string winName = "image";
	// cv::imshow(winName, h_image);
	// cv::waitKey(0);
	// cv::destroyWindow(winName);

    return hwgb_image;
}

Mat apc_object::calcHistogram(Mat image, Mat mask, int range_low, int rangle_high)
{
	Mat hist;
	int hbins = rangle_high - range_low + 1;
    int histSize[] = {hbins};
    float hranges[] = { range_low, rangle_high };
    const float* ranges[] = { hranges };
    int channels[] = {0};
    calcHist( &image, 1, channels, mask,
             hist, 1, histSize, ranges,
             true,
             false );
    return hist;
}

double apc_object::getColorProbability(int feature)
	{return ColorHist.at<float>(feature,1);}

double apc_object::getHeightProbability(int feature)
	{return HeightHist.at<float>(feature,1);}

double apc_object::getNormalProbability(int feature)
	{return NormalHist.at<float>(feature,1);}

void apc_object::ReadHistograms(){
	std::string filename = "ColorHist"+ patch::to_string(label)+".txt";
	readHistogramtoFile(filename, ColorHist);
	std::cout<<ColorHist<<std::endl;
	std::cout<<"*******************************"<<std::endl;

	// filename = "HeightHist"+ patch::to_string(label)+".txt";
	// readHistogramtoFile(filename, HeightHist);
	// std::cout<<HeightHist<<std::endl;
	// std::cout<<"*******************************"<<std::endl;

	filename = "NormalHist"+ patch::to_string(label)+".txt";
	readHistogramtoFile(filename, NormalHist);
	std::cout<<NormalHist<<std::endl;
	std::cout<<"*******************************"<<std::endl;
}
