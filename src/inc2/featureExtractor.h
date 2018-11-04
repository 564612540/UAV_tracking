#pragma once
#include <opencv2/opencv.hpp>
#include <vector>

#include "motion.h"
#include "gradientMex.h"
#include "parameters.h"
#include "fhog.hpp"

class HogRawPixelNormExtractor
{
private:
	cv::Mat rawFeat;
	cv::Mat hogFeat;
	cv::HOGDescriptor hog;
	int dims;
public:
	HogRawPixelNormExtractor();
	cv::Mat RawPixel(const cv::Mat &img, const cv::Mat &oriRects);
	cv::Mat HOG(const cv::Mat &img, const cv::Mat &rects);
	cv::Mat extract(const cv::Mat &img, const cv::Mat &rects);
	void extract(const cv::Mat &img, const cv::Mat &rects, cv::Mat &feat);
	cv::Mat computeFhog(const cv::Mat &img, const cv::Mat &oriRects);

	//img传入原始图像
	cv::Mat FHOG2(const cv::Mat &img, const cv::Mat &oriRects);
};

cv::Mat	multiThr_extract(HogRawPixelNormExtractor &hne, const cv::Mat &img, const cv::Mat &rects);
void multiThr_extractCopy(HogRawPixelNormExtractor &hne, const cv::Mat &img, const cv::Mat &rects, cv::Mat &feat);