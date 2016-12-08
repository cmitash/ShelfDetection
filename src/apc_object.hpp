#ifndef APC_OBJECT
#define APC_OBJECT

#include "sensing_common.h"
#include "apc_shelf.hpp"

using namespace cv;

/*Enumerated Lables for each obect*/
enum ObjectLabels {
	PRX_NONE,
	PRX_DUMBBELLS,
	PRX_PAPER_TOWEL,
	PRX_CLEAN_BRUSH,
	PRX_BUBBLE_MAILER,
	PRX_COMMAND_HANG,
	PRX_STEMS,
	PRX_CURTAIN_LINER,
	PRX_SOCKS,
	PRX_PLUG_PROTECTOR
};

#define HUE_WHITE 180
#define HUE_GRAY 181
#define HUE_BLACK 182

/*Class to maintain information about objects used in apc*/
class apc_object
{
	public:
		apc_object();
		apc_object(ObjectLabels label, std::string, tf::StampedTransform, std::vector<int>);
		~apc_object();

		void LoadData();
		void ShowTrainingImages();
		void ConvertHSVtoHWGB();
		cv::Mat ConvertImgtoHWGB(Mat image);
		cv::Mat calcHistogram(Mat image, Mat mask, int range_low, int rangle_high);
		void getColorFeatures();
		cv::Mat ConvertImgtoHeightImg(Mat, double);
		void getShelfHeightFeature(apc_shelf*);
		void getNormalCurvature();
		Mat ConvertImgtoNormalImg(Mat image, Mat mask);
		double getColorProbability(int);
		double getHeightProbability(int);
		double getNormalProbability(int);
		void ReadHistograms();
	protected:
		ObjectLabels label;
		std::vector<cv::Mat> TrainingMask;
		std::vector<std::string> TrainingImgNames;
		std::vector<int> bin;
		std::string path;

		cv::Mat ColorHist;
		cv::Mat HeightHist;
		cv::Mat NormalHist;
		cv::Mat NaNHist;
		tf::StampedTransform CameraToWorldTf;
};

#endif