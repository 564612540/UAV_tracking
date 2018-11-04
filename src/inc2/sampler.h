#pragma once
#include "parameters.h"
#include "motion.h"
#include <opencv2/opencv.hpp>

using namespace cv;
class SlidingWindowSampler
{
private:
	int pos_wlength;
	int pos_hlength;
	Mat pos_w;
	Mat pos_h;
	Mat pos_res;

private:
	int neg_wlength;
	int neg_hlength;
	Mat neg_w;
	Mat neg_h;
	Mat neg_res;

public:
	SlidingWindowSampler();
	Mat doPosSample(const Mat target);
	Mat doNegSample(const Mat target);
	Mat quickPos(const Mat &target);
	Mat quickNeg(const Mat &target);
};