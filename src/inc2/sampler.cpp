#include "sampler.h"

SlidingWindowSampler::SlidingWindowSampler()
{
	{
		pos_wlength = POS_SLIDING_W + 1, pos_hlength = POS_SLIDING_H + 1;
		int *ptr_w = 0, *ptr_h = 0;
		pos_w = Mat(1, pos_wlength, CV_32SC1);
		pos_h = Mat(pos_hlength, 1, CV_32SC1);
		ptr_w = (int*)pos_w.data;
		ptr_h = (int*)pos_h.data;
		for (int i = 0; i < pos_wlength; i++)
		{
			ptr_w[i] = i - round(POS_SLIDING_W / 2.0);
		}
		for (int i = 0; i < pos_hlength; i++)
		{
			ptr_h[i] = i - round(POS_SLIDING_H / 2.0);
		}
		pos_w = repeat(pos_w, pos_hlength, 1);
		pos_h = repeat(pos_h, 1, pos_wlength);
		pos_res = Mat(pos_wlength*pos_hlength + 1, 4, CV_32FC1);
	}

	{
		neg_wlength = NEG_SLIDING_W / NEG_SRIDE + 1, neg_hlength = NEG_SLIDING_H / NEG_SRIDE + 1;
		neg_w = Mat(1, neg_wlength, CV_32SC1);
		neg_h = Mat(neg_hlength, 1, CV_32SC1);
		int *ptr_w = 0, *ptr_h = 0;

		ptr_w = (int*)neg_w.data;
		ptr_h = (int*)neg_h.data;
		for (int i = 0, j = 0; i <= NEG_SLIDING_W; i += NEG_SRIDE)
		{
			ptr_w[j] = i - round(NEG_SLIDING_W / 2.0);
			j++;
		}
		for (int i = 0, j = 0; i <= NEG_SLIDING_H; i += NEG_SRIDE)
		{
			ptr_h[j] = i - round(NEG_SLIDING_H / 2.0);
			j++;
		}

		neg_w = repeat(neg_w, neg_hlength, 1);
		neg_h = repeat(neg_h, 1, neg_wlength);
		neg_res = Mat(neg_wlength*neg_hlength + 1, 4, CV_32FC1);
		
	}
}

Mat SlidingWindowSampler::doPosSample(const Mat target)
{
	int wlength = POS_SLIDING_W + 1, hlength = POS_SLIDING_H + 1;
	int *ptr_w = 0, *ptr_h = 0;
	Mat w(1, wlength, CV_32SC1);
	Mat h(hlength, 1, CV_32SC1);
	ptr_w = (int*)w.data;
	ptr_h = (int*)h.data;
	for (int i = 0; i < wlength; i++)
	{
		ptr_w[i] = i - round(POS_SLIDING_W / 2.0);
		//w.at<int>(i) = i - round(POS_SLIDING_W / 2.0);		//at可以换成更高效的指针访问
	}
	for (int i = 0; i < hlength; i++)
	{
		ptr_h[i] = i - round(POS_SLIDING_H / 2.0);
		//h.at<int>(i) = i - round(POS_SLIDING_H / 2.0);
	}
	//std::cout << "w:" << w << std::endl;
	//std::cout << "h:" << h << std::endl;
	w = repeat(w, hlength, 1);
	h = repeat(h, 1, wlength);
	ptr_w = (int*)w.data;
	ptr_h = (int*)h.data;
	Mat res = repeat(target, wlength*hlength+1, 1);
	for (int i = 0; i < wlength*hlength; i++)
	{
		res.at<float>(i, 0) += ptr_w[i];
		res.at<float>(i, 1) += ptr_h[i];
		/*res.at<float>(i, 0) += w.at<int>(i);
		res.at<float>(i, 1) += h.at<int>(i);*/
	}
	//std::cout << res << std::endl;
	return res;
}

Mat SlidingWindowSampler::doNegSample(const Mat target)
{
	int wlength = NEG_SLIDING_W / NEG_SRIDE + 1, hlength = NEG_SLIDING_H / NEG_SRIDE + 1;
	Mat w(1, wlength, CV_32SC1);
	Mat h(hlength, 1, CV_32SC1);
	int *ptr_w = 0, *ptr_h = 0;

	ptr_w = (int*)w.data;
	ptr_h = (int*)h.data;
	for (int i = 0, j=0; i <= NEG_SLIDING_W; i+=NEG_SRIDE)
	{
		ptr_w[j] = i - round(NEG_SLIDING_W / 2.0);
		j++;
	}
	for (int i = 0, j=0; i <= NEG_SLIDING_H; i += NEG_SRIDE)
	{
		ptr_h[j] = i - round(NEG_SLIDING_H / 2.0);
		j++;
	}
	//std::cout << "w:" << w << std::endl;
	//std::cout << "h:" << h << std::endl;
	w = repeat(w, hlength, 1);
	h = repeat(h, 1, wlength);
	ptr_w = (int*)w.data;
	ptr_h = (int*)h.data;

	Mat res = repeat(target, wlength*hlength + 1, 1);
	Mat Idx;
	//std::cout << res;
	for (int i = 0; i < wlength*hlength; i++)
	{
		res.at<float>(i, 0) += ptr_w[i];
		//std::cout << res.at<float>(i, 0) << std::endl;
		res.at<float>(i, 1) += ptr_h[i];
		//std::cout << res.at<float>(i, 1) << std::endl;
		/*float x, y, z, p;
		x = abs(w.at<int>(i));
		y = abs(h.at<int>(i));
		z = res.at<float>(i, 3)*NEG_EXCLUDE_RATIO;
		p = res.at<float>(i, 2)*NEG_EXCLUDE_RATIO;
		std::cout << x<<" " << y<< " " << z<<" " << p<<std::endl;*/
		if ( (abs(ptr_w[i]) > (res.at<float>(i, 2)*NEG_EXCLUDE_RATIO)) || (abs(ptr_h[i]) > (res.at<float>(i, 3)*NEG_EXCLUDE_RATIO)) )
		{
			Idx.push_back(i);
		}
	}
	//std::cout << "res" << res << std::endl;
	//std::cout << "Idx" << Idx << std::endl;
	res = getRows(res, Idx);
	return res;
}

Mat SlidingWindowSampler::quickPos(const Mat & target)
{
	int* ptr_w = (int*)pos_w.data;
	int* ptr_h = (int*)pos_h.data;
	
	float* ptr_res = (float*)pos_res.data;
	float* ptr_tar = (float*)target.data;

	for (int i = 0, j = 0; i < pos_res.rows; i++, j += 4)
	{
		ptr_res[j + 0] = ptr_tar[0];
		ptr_res[j + 1] = ptr_tar[1];
		ptr_res[j + 2] = ptr_tar[2];
		ptr_res[j + 3] = ptr_tar[3];
	}

	for (int i = 0; i < pos_wlength*pos_hlength; i++, ptr_res += 4)
	{
		ptr_res[0] += ptr_w[i];
		ptr_res[1] += ptr_h[i];
	}
	return pos_res;
}

Mat SlidingWindowSampler::quickNeg(const Mat & target)
{

	int* ptr_w = (int*)neg_w.data;
	int* ptr_h = (int*)neg_h.data;

	float* ptr_res = (float*)neg_res.data;
	float* ptr_tar = (float*)target.data;

	for (int i = 0, j=0; i < neg_res.rows; i++, j+=4)
	{
		ptr_res[j + 0] = ptr_tar[0];
		ptr_res[j + 1] = ptr_tar[1];
		ptr_res[j + 2] = ptr_tar[2];
		ptr_res[j + 3] = ptr_tar[3];
	}

	Mat Idx;
	for (int i = 0; i < neg_wlength*neg_hlength; i++, ptr_res+=4)
	{
		ptr_res[0] += ptr_w[i];
		ptr_res[1] += ptr_h[i];
		
		if ((abs(ptr_w[i]) > (ptr_res[2]*NEG_EXCLUDE_RATIO)) || (abs(ptr_h[i]) > (ptr_res[3]*NEG_EXCLUDE_RATIO)))
		{
			Idx.push_back(i);
		}
	}
	return getRows(neg_res, Idx);
}
