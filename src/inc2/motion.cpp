#include "motion.h"
#include <thread>
using namespace std;
using namespace cv;
//各个函数还需测试，主要测试clone，copyto这些函数用法是否正确。

ParticleFilterMotionModel::ParticleFilterMotionModel()
{
	rands = cv::Mat(PARTICLE_N, 4, CV_32FC1);
}

void ParticleFilterMotionModel::genParticle(cv::Mat &rect, cv::Mat &conf)
{
	double minVal = 0, maxVal = 0;
	int minIdx[2] = { 0 }, maxIdx[2] = { 0 };
	cv::Mat conftmp = conf.clone();
	conftmp = conftmp.t();
	int N = PARTICLE_N;
	cv::minMaxIdx(conftmp, &minVal, NULL, NULL, maxIdx);
	cv::Mat maxConfRow = rect.row(maxIdx[0]).clone();
	if (rect.rows == 1)
	{
		rect = cv::repeat(rect, PARTICLE_N, 1);
	}
	else
	{
		N = conftmp.rows;
		cv::Mat nRands(cv::Size(N, 1), CV_32FC1);
		conftmp = conftmp - minVal;
		cv::exp(conftmp / PARTICLE_CONDENSE, conftmp);
		conftmp=  conftmp / cv::sum(conftmp)[0];
		cumsum(conftmp);
		conftmp = cv::repeat(conftmp, 1, N);
		
		RNG rng(20);
		rng.fill(nRands, cv::RNG::UNIFORM, 0, 1, true);
		/*cv::FileStorage fs("./nRands.xml", cv::FileStorage::WRITE);
		fs << "nRands" << nRands;
		fs.release();*/
		nRands = cv::repeat(nRands, N, 1);
		cv::Mat_<int> Idx;
		sumVector((nRands > conftmp)/255, 0).convertTo(Idx, CV_32SC1);
		rect = getRows(rect, Idx);
	}
	
	int ratio[2] = HOG_WIN;
	float affsig[4]=PARTICLE_AFFSIG;
	cv::Mat_<float> multier(1,4,CV_32FC1);
	float* mul=(float*)multier.data;
	mul[0]=affsig[0];
	mul[1]=affsig[1];
	mul[2]=affsig[2];
	mul[3]=affsig[3];
	multier = cv::repeat(multier, N, 1);
	
	RNG rng(20);
	cv::Mat randn(N, 4, CV_32FC1);
	rng.fill(randn, cv::RNG::NORMAL, 0, 1);//这个函数可能有毒，需要多多关注
	/*cv::FileStorage fs("randn.xml", cv::FileStorage::WRITE);
	fs << "randn" << randn;
	fs.release();*/
	
	rect.col(3) = rect.col(3) / rect.col(2);
	rect.col(2) = rect.col(2) / ratio[0];
	
	/*cout << randn << endl;
	cout << multier << endl;*/
	std::cout<<multier.rows<<" "<<multier.cols;
	
	rect = rect + randn.mul(multier);
	
	rect.col(2) = rect.col(2) * ratio[0];
	rect.col(3) = rect.col(3).mul(rect.col(2));
	
	cv::Mat result(rect.rows, rect.cols, rect.type());
	randperm(rect, result);
	maxConfRow.copyTo(result.row(0));
	filter(result);
	rect = result;
}

cv::Mat sumVector(cv::Mat src, int dir)
{
	cv::Mat res;
	if (dir)
	{
		res = cv::Mat(src.rows, 1, CV_64FC1);
		for (int i = 0; i < src.rows; i++)
		{
			res.at<double>(i) = cv::sum(src.row(i))[0];
		}
	}
	else
	{
		res = cv::Mat(1, src.cols, CV_64FC1);
		for (int i = 0; i < src.cols; i++)
		{
			res.at<double>(i) = cv::sum(src.col(i))[0];
		}

	}
	return res;
}
cv::Mat getRows(cv::Mat &src, cv::Mat Idx)
{
	cv::Mat res(Idx.rows * Idx.cols, src.cols, CV_32FC1);
	for (int i = 0; i < Idx.rows; i++)
	{
		for (int j = 0; j < Idx.cols; j++)
		{
			int index = Idx.at<int>(i, j);
			src.row(index).copyTo(res.row(i*Idx.cols + j));
		}
	}
	return res;
}
void randperm(cv::Mat &src, cv::Mat &dst)
{
	std::vector<int> shuffle(src.rows);
	for (int i = 0; i < shuffle.size(); i++)
	{
		shuffle[i] = i;
	}
	
	std::shuffle(shuffle.begin(), shuffle.end(), std::default_random_engine(time(0)));
	for (int i = 0; i < shuffle.size(); i++)
	{
		src.row(shuffle[i]).copyTo(dst.row(i));
	}
	/*std::cout << src << std::endl;
	std::cout << dst << std::endl;*/
}

void filter(cv::Mat & src)
{
	int tailRow = src.rows;
	int i = 0;
	while( i!=tailRow)
	{
		if (src.at<float>(i, 3) <= 3 || src.at<float>(i, 2) <= 3)
		{
			tailRow--;
			src.row(tailRow).copyTo(src.row(i));
		}
		else
		{
			i++;
		}
	}
	if (src.rows != tailRow)
	{
		src.pop_back(src.rows - tailRow);
	}
}

inline void cumsum(cv::Mat &src)
{
	cumsum(src, src);
}

inline void cumsum(cv::Mat &src, cv::Mat &dst)
{
	int rows = src.rows;
	dst.row(0) = src.row(0);
	for (int i = 1; i < rows; i++)
	{
		dst.row(i) = src.row(i) + dst.row(i - 1);
	}
}
