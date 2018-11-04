#pragma once

#include "parameters.h"
#include <vector>
#include <opencv2/opencv.hpp>
#include <vector>
#include <algorithm>
#include <random>

class ParticleFilterMotionModel
{
private:
	cv::Mat rands;
public:
	ParticleFilterMotionModel();
	void genParticle(cv::Mat &rect, cv::Mat &conf); //根据上一帧预估位置生成粒子
};

inline void cumsum(cv::Mat &src);

inline void cumsum(cv::Mat &src, cv::Mat &dst);

cv::Mat sumVector(cv::Mat src, int dir);

cv::Mat getRows(cv::Mat &src, cv::Mat Idx);

void randperm(cv::Mat &src, cv::Mat &dst);

void filter(cv::Mat &src);