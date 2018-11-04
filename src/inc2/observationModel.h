#pragma once
#include <opencv2/opencv.hpp> 
#include "parameters.h"

class LogisticRegression
{
private:
	cv::Mat weight;
	int num;
	
public:
	int flag;
	LogisticRegression();
	void trainLR(const cv::Mat &posFeat, const cv::Mat &negFeat);
	inline bool isEmpty()
	{
		return weight.empty();
	}
	inline cv::Mat LR(cv::Mat feat)
	{
		cv::Mat res = weight.colRange(0, num - 1)*feat + weight.at<float>(num - 1);
		exp(-res, res);
		res = 1 / (1 + res);
		return res;
	}
};
void trainLR_Thr(LogisticRegression *lr,const cv::Mat &posFeat, const cv::Mat &negFeat);
