#pragma once
#include <opencv2/opencv.hpp>
#include <sys/types.h>
#include <sys/shm.h>
#include <sys/sem.h>

//#include<opencv2/core/core.hpp>
#include "featureExtractor.h"
#include "motion.h"
#include "observationModel.h"
#include "parameters.h"
#include "sampler.h"
#include "threadpool.h"
#include "updater.h"
#include "threadpool.h"

using namespace cv;
using namespace std;

class StapleTracker
{
private:
	HogRawPixelNormExtractor extractor;
	LogisticRegression lr;
	SlidingWindowSampler sampler;
	ParticleFilterMotionModel motion;
	ClassificationScoreJudger judger;
	threadpool *worker;
private:
	cv::Mat rects;
	int maxIdx[2];
	double maxProb;

	Mat posSample, negSample;
	Mat posFeature, negFeature, feat;
	Mat probs;
	Mat img;

	future<void> trainF;
public:
	StapleTracker();
	~StapleTracker();
	void init(Mat image, Mat rect);
	bool init(Mat image, vector<int> bounding_box);
	bool track(Mat image);
	bool update(Mat image);
	vector<float> position();
	bool drawCurrentImg(Mat image);
};

struct transData{
	bool start;
	bool isLost;
	float posi[3];
	int bounding_box[4];
	Mat image;
};

union semun  
{  
    int val;  
    struct semid_ds *buf;  
    unsigned short *arry;  
};
static int sem_id;
static int set_semvalue();  
static void del_semvalue();  
static int semaphore_p();  
static int semaphore_v();
