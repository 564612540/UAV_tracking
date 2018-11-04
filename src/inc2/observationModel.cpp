#include "observationModel.h"


using namespace cv;
using namespace std;

LogisticRegression::LogisticRegression()
{
	flag = 0;
}

void LogisticRegression::trainLR(const cv::Mat & posFeat,const cv::Mat & negFeat)
{
	clock_t start = clock(), finish;
	int maxIter, totalNum=posFeat.cols+negFeat.cols;
	float lambda = 1e-2;
	float lr = 1.0 / totalNum;
	float alpha = 0.99;
	
	if (weight.empty())
	{
		RNG rng;
		weight = cv::Mat(1, posFeat.rows + 1, CV_32FC1);
		rng.fill(weight, RNG::NORMAL, 0, 1);
		num = weight.cols;
		/*FileStorage fs("./weight.xml", FileStorage::WRITE);
		fs << "weight" << weight;
		fs.release();*/
		weight /= weight.cols;
		maxIter = COLDBOOT_ITER;
		
	}
	else
	{
		maxIter = ITER;
	}
	
	Mat feat(posFeat.rows, totalNum, posFeat.type()), featWithBias;
	Mat label = Mat::zeros(1, totalNum, feat.type());
	Mat deltaw = Mat::zeros(1,weight.cols, weight.type());
	posFeat.copyTo(feat.colRange(0, posFeat.cols));
	negFeat.copyTo(feat.colRange(posFeat.cols, totalNum));

	Mat tmp = Mat::ones(1,feat.cols,feat.type());
	feat.push_back(tmp);
	
	featWithBias = feat;
	//featWithBias.push_back(Mat::ones(1, featWithBias.cols, featWithBias.type()));//bias多余，合并再一起，加进去取出一部分来即可，再把变量声明整合一下。
	feat = featWithBias.rowRange(0, num - 1);
	
	featWithBias = featWithBias.t();
	
	label.colRange(0, posFeat.cols) += 1;
	//std::cout << label << std::endl;
	
	Mat loss;
	for (int i = 0; i < maxIter; i++)
	{
		start = clock();
		loss = label - LR(feat);
		deltaw = (loss * featWithBias - lambda*weight)*lr + alpha*deltaw;
		weight += deltaw;
		finish = clock();
		//std::cout <<"trained one iteration time::"<< (double)(finish - start) / CLOCKS_PER_SEC;
	}
	finish = clock();
	std::cout << "\ntrained time::" << (double)(finish - start) / CLOCKS_PER_SEC;
}

void trainLR_Thr(LogisticRegression * lr, const cv::Mat & posFeat, const cv::Mat & negFeat)
{
	lr->trainLR(posFeat, negFeat);
}
