#include "featureExtractor.h"
#include <vector>
#include<iostream>
using namespace cv;
//矩形目前数据格式x,y坐标,宽,高.
HogRawPixelNormExtractor::HogRawPixelNormExtractor()
{
	
}

cv::Mat HogRawPixelNormExtractor::RawPixel(const cv::Mat & img, const cv::Mat & oriRects)
{
	Mat new_img = img.clone();
	Mat rects = oriRects.clone();
	//待优化
	new_img.convertTo(new_img, CV_32FC3, 1. / 255);
	int shift = 0;
	if (3 == img.channels())
	{
		cvtColor(new_img, new_img, CV_BGR2Lab);
	}
	double minx = 0, miny = 0;
	double maxx = 0, maxy = 0;
	minMaxLoc(rects.col(0), &minx);
	minMaxLoc(rects.col(1), &miny);
	if (round(minx) < 0)
	{
		shift = abs(round(minx));
		Mat tmp = Mat::zeros(new_img.rows, new_img.cols + shift, new_img.type());
		new_img.copyTo(tmp.colRange(shift, tmp.cols));
		new_img = tmp;
		rects.col(0) += shift;
	}
	if (round(miny) < 0)
	{
		shift = abs(round(miny));
		Mat tmp = Mat::zeros(new_img.rows + shift, new_img.cols, new_img.type());
		new_img.copyTo(tmp.rowRange(shift, tmp.rows));
		new_img = tmp;
		rects.col(1) += shift;
	}

	minMaxLoc(rects.col(0) + rects.col(2), (double *)0, &maxx);
	minMaxLoc(rects.col(1) + rects.col(3), (double *)0, &maxy);

	if (round(maxx)>new_img.cols)
	{
		shift = round(maxx) - new_img.cols;
		Mat tmp = Mat::zeros(new_img.rows, new_img.cols + shift, new_img.type());
		new_img.copyTo(tmp.colRange(0, new_img.cols));
		new_img = tmp;
	}
	if (round(maxy) > new_img.rows)
	{
		shift = round(maxy) - new_img.rows;
		Mat tmp = Mat::zeros(new_img.rows + shift, new_img.cols, new_img.type());
		new_img.copyTo(tmp.rowRange(0, new_img.rows));
		new_img = tmp;
	}
	int size[2] = HOG_WIN;
	Mat res(size[0] * size[1] * new_img.channels()/4, rects.rows, CV_32FC1);

	for (int i = 0; i < rects.rows; i++)
	{
		Mat tmp;
		resize(new_img(Rect(rects.at<float>(i,0), rects.at<float>(i,1), rects.at<float>(i,2), rects.at<float>(i,3))), tmp, Size(size[0]/2, size[1]/2));
		tmp = tmp.reshape(1, res.rows);
		tmp.copyTo(res.col(i));
		
		double normalization = norm(res.col(i));
		if( normalization> 1e-6)
		{
			res.col(i) /= normalization;
		}
		
	}
	return res;
}

cv::Mat HogRawPixelNormExtractor::HOG(const cv::Mat & img, const cv::Mat & rects)
{
	
}

cv::Mat HogRawPixelNormExtractor::extract(const cv::Mat & img, const cv::Mat & rects)
{
	//传入一个参数接收特征计算结果，免去结果复制开销
	clock_t start = clock();
	//hogFeat = HOG(img, rects);
	hogFeat = computeFhog(img, rects);
	//hogFeat = FHOG2(img, rects);
	rawFeat = RawPixel(img, rects);
	Mat res(hogFeat.rows + rawFeat.rows, rawFeat.cols, hogFeat.type());
	hogFeat.copyTo(res.rowRange(0, hogFeat.rows));
	rawFeat.copyTo(res.rowRange(hogFeat.rows, res.rows));
	Mat tmp = res.mul(res);
	tmp = sumVector(tmp, 0);
	sqrt(tmp, tmp);
	for (int i = 0; i < tmp.cols; i++)
	{
		res.col(i) /= tmp.at<double>(i);
	}
	clock_t finish = clock();
	double time = (double)(finish - start) / CLOCKS_PER_SEC;
	printf("\ntime cost: %f seconds", time);
	return res;
}

void HogRawPixelNormExtractor::extract(const cv::Mat & img, const cv::Mat & rects, cv::Mat & feat)
{
	//传入一个参数接收特征计算结果，免去结果复制开销
	clock_t start = clock();
	//hogFeat = HOG(img, rects);
	hogFeat = computeFhog(img, rects);
	//hogFeat = FHOG2(img, rects);
	rawFeat = RawPixel(img, rects);
	hogFeat.copyTo(feat.rowRange(0, hogFeat.rows));
	rawFeat.copyTo(feat.rowRange(hogFeat.rows, feat.rows));
	Mat tmp = feat.mul(feat);
	tmp = sumVector(tmp, 0);
	sqrt(tmp, tmp);
	for (int i = 0; i < tmp.cols; i++)
	{
		feat.col(i) /= tmp.at<double>(i);
	}
}

cv::Mat HogRawPixelNormExtractor::computeFhog(const cv::Mat & img, const cv::Mat &oriRects)
{
	Mat new_img = img.clone();
	Mat rects = oriRects.clone();
	int shift;
	double minx, miny;
	double maxx, maxy;
	
	minMaxLoc(rects.col(0), &minx);
	minMaxLoc(rects.col(1), &miny);
	
	if (round(minx) < 0)
	{
		shift = abs(round(minx) - 1);
		Mat tmp = Mat::zeros(new_img.rows, new_img.cols + shift, new_img.type());
		new_img.copyTo(tmp.colRange(shift, tmp.cols));
		new_img = tmp;
		rects.col(0) += shift;
	}
	if (round(miny) < 0)
	{
		shift = abs(round(miny) - 1);
		Mat tmp = Mat::zeros(new_img.rows + shift, new_img.cols, new_img.type());
		new_img.copyTo(tmp.rowRange(shift, tmp.rows));
		new_img = tmp;
		rects.col(1) += shift;
	}

	minMaxLoc(rects.col(0) + rects.col(2), NULL, &maxx);
	minMaxLoc(rects.col(1) + rects.col(3), NULL, &maxy);

	if (round(maxx)>new_img.cols)
	{
		shift = round(maxx) - new_img.cols;
		Mat tmp = Mat::zeros(new_img.rows, new_img.cols + shift, new_img.type());
		new_img.copyTo(tmp.colRange(0, new_img.cols));
		new_img = tmp;
	}
	if (round(maxy) > new_img.rows)
	{
		shift = round(maxy) - new_img.rows;
		Mat tmp = Mat::zeros(new_img.rows + shift, new_img.cols, new_img.type());
		new_img.copyTo(tmp.rowRange(0, new_img.rows));
		new_img = tmp;
	}
	
	int numOflayer = 1024;
	float *pixels = new float[numOflayer * new_img.channels()];
	Mat res(512, rects.rows, CV_32FC1);
	float h[512] = { 0 };
	float m[1024] , o[1024];
	Mat hist(512, 1, CV_32FC1, h);
	
	for (int i = 0; i < rects.rows; i++) 
	{
		Rect rect(rects.at<float>(i, 0), rects.at<float>(i, 1), rects.at<float>(i, 2), rects.at<float>(i, 3));
		Mat itmp = new_img(rect).clone();
		//std::cout << rects << std::endl;
		resize(itmp, itmp, Size(32, 32));
		for (int i = 0; i < itmp.rows; i++)
		{
			for (int j = 0; j < itmp.cols; j++)
			{
				Vec3b tmp = itmp.at<Vec3b>(i, j);
				int pos = i*itmp.rows + j;
				pixels[pos] = tmp[2] / 255.f;
				pixels[pos + numOflayer] = tmp[1] / 255.f;
				pixels[pos + 2 * numOflayer] = tmp[0] / 255.f;
			}
		}
		gradMag(pixels, m, o, 32, 32, 3, 1);
		fhog(m, o, h, 32, 32, 8, 9, -1, 0.2f);
		hist /= norm(hist);
		hist.copyTo(res.col(i));
	}
	delete[]pixels;
	return res;
}

cv::Mat HogRawPixelNormExtractor::FHOG2(const cv::Mat & img, const cv::Mat & oriRects)
{
	Mat new_img = img.clone();
	Mat rects = oriRects.clone();
	int shift;
	double minx, miny;
	double maxx, maxy;

	minMaxLoc(rects.col(0), &minx);
	minMaxLoc(rects.col(1), &miny);

	if (round(minx) < 0)
	{
		shift = abs(round(minx) - 1);
		Mat tmp = Mat::zeros(new_img.rows, new_img.cols + shift, new_img.type());
		new_img.copyTo(tmp.colRange(shift, tmp.cols));
		new_img = tmp;
		rects.col(0) += shift;
	}
	if (round(miny) < 0)
	{
		shift = abs(round(miny) - 1);
		Mat tmp = Mat::zeros(new_img.rows + shift, new_img.cols, new_img.type());
		new_img.copyTo(tmp.rowRange(shift, tmp.rows));
		new_img = tmp;
		rects.col(1) += shift;
	}
	minMaxLoc(rects.col(0) + rects.col(2), NULL, &maxx);
	minMaxLoc(rects.col(1) + rects.col(3), NULL, &maxy);
	if (round(maxx)>new_img.cols)
	{
		shift = round(maxx) - new_img.cols;
		Mat tmp = Mat::zeros(new_img.rows, new_img.cols + shift, new_img.type());
		new_img.copyTo(tmp.colRange(0, new_img.cols));
		new_img = tmp;
	}
	if (round(maxy) > new_img.rows)
	{
		shift = round(maxy) - new_img.rows;
		Mat tmp = Mat::zeros(new_img.rows + shift, new_img.cols, new_img.type());
		new_img.copyTo(tmp.rowRange(0, new_img.rows));
		new_img = tmp;
	}
	Mat res, tmp;
	int win[2] = HOG_WIN;
	for (int i = 0; i < rects.rows; i++)
	{
		Rect rect(rects.at<float>(i,0),rects.at<float>(i,1), rects.at<float>(i,2), rects.at<float>(i,3));
		tmp = new_img(rect).clone();
		tmp.convertTo(tmp, CV_32FC1, 1 / 255.f);
		//std::cout << tmp << std::endl;
		cvtColor(tmp, tmp, CV_BGR2GRAY);
		resize(tmp, tmp, Size(win[0], win[1]));
		Mat descriptor = FHoG::extract(tmp, 2, 8);
		if (i == 0)
		{
			res = Mat(descriptor.rows, rects.rows, CV_32FC1);
			//cout << descriptor << endl;
		}
		descriptor.copyTo(res.col(i));
	}
	return res;
}

cv::Mat multiThr_extract(HogRawPixelNormExtractor &hne, const cv::Mat & img, const cv::Mat & rects)
{
	return hne.extract(img, rects);
}

void multiThr_extractCopy(HogRawPixelNormExtractor & hne, const cv::Mat & img, const cv::Mat & rects, cv::Mat & feat)
{
	hne.extract(img, rects, feat);
}

cv::Mat test(const cv::Mat &img, const cv::Mat &oriRects)
{
	Mat new_img = img.clone();
	Mat rects = oriRects.clone();
	int shift;
	double minx, miny;
	double maxx, maxy;

	minMaxLoc(rects.col(0), &minx);
	minMaxLoc(rects.col(1), &miny);

	if (round(minx) < 0)
	{
		shift = abs(round(minx) - 1);
		Mat tmp = Mat::zeros(new_img.rows, new_img.cols + shift, new_img.type());
		new_img.copyTo(tmp.colRange(shift, tmp.cols));
		new_img = tmp;
		rects.col(0) += shift;
	}
	if (round(miny) < 0)
	{
		shift = abs(round(miny) - 1);
		Mat tmp = Mat::zeros(new_img.rows + shift, new_img.cols, new_img.type());
		new_img.copyTo(tmp.rowRange(shift, tmp.rows));
		new_img = tmp;
		rects.col(1) += shift;
	}

	minMaxLoc(rects.col(0) + rects.col(2), NULL, &maxx);
	minMaxLoc(rects.col(1) + rects.col(3), NULL, &maxy);

	if (round(maxx)>new_img.cols)
	{
		shift = round(maxx) - new_img.cols;
		Mat tmp = Mat::zeros(new_img.rows, new_img.cols + shift, new_img.type());
		new_img.copyTo(tmp.colRange(0, new_img.cols));
		new_img = tmp;
	}
	if (round(maxy) > new_img.rows)
	{
		shift = round(maxy) - new_img.rows;
		Mat tmp = Mat::zeros(new_img.rows + shift, new_img.cols, new_img.type());
		new_img.copyTo(tmp.rowRange(0, new_img.rows));
		new_img = tmp;
	}

	int numOflayer = 1024;
	float *pixels = new float[numOflayer * new_img.channels()];
	Mat res(512, rects.rows, CV_32FC1);
	float h[512] = { 0 };
	float m[1024], o[1024];
	Mat hist(512, 1, CV_32FC1, h);

	for (int i = 0; i < rects.rows; i++)
	{
		Rect rect(rects.at<float>(i, 0), rects.at<float>(i, 1), rects.at<float>(i, 2), rects.at<float>(i, 3));
		Mat itmp = new_img(rect).clone();
		//std::cout << rects << std::endl;
		resize(itmp, itmp, Size(32, 32));

		for (int i = 0; i < itmp.rows; i++)
		{
			for (int j = 0; j < itmp.cols; j++)
			{
				Vec3b tmp = itmp.at<Vec3b>(i, j);
				int pos = i*itmp.rows + j;
				pixels[pos] = tmp[2] / 255.f;
				pixels[pos + numOflayer] = tmp[1] / 255.f;
				pixels[pos + 2 * numOflayer] = tmp[0] / 255.f;
			}
		}
		gradMag(pixels, m, o, 32, 32, 3, 1);
		fhog(m, o, h, 32, 32, 8, 9, -1, 0.2f);
		hist /= norm(hist);
		hist.copyTo(res.col(i));
	}
	delete[]pixels;
	return res;
}
