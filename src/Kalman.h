#pragma once
#include<iostream>
#include<vector>
#include<cassert>
#include<opencv2/core/core.hpp>
//#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

class Kalman {
private:
	vector<float> pos;
	vector<float> status;
	float distance;
	typedef struct {
		float fx;
		float fy;
		float tx;
		float ty;
		float ts;
		float qk;
		float rk;
	}params;
	params para_;
	Mat pk;
	Mat rk, qk;
	vector<float> f_x(vector<float> u_k) {
		assert(u_k.size() == 4);
		vector<float> result(6);
		result[0]=cos(para_.ts*u_k[3])*(status[0] + para_.ts*status[1]) - sin(para_.ts*u_k[3])*(status[4] + para_.ts*status[5]) + para_.ts*u_k[1];
		result[1]=cos(para_.ts*u_k[3])*status[1] - sin(para_.ts*u_k[3])*status[5] + u_k[1] - u_k[3]*result[0];
		result[2]=status[2] + para_.ts*status[3] + u_k[2] * para_.ts;
		result[3]=status[3] + u_k[2];
		result[4]=sin(para_.ts*u_k[3])*(status[0] + para_.ts*status[1]) + cos(para_.ts*u_k[3])*(status[4] + para_.ts*status[5]) - para_.ts*u_k[0];
		result[5]=sin(para_.ts*u_k[3])*status[1] + cos(para_.ts*u_k[3])*status[5] - u_k[0] - u_k[3] * result[4];
		return result;
	}
	vector<float> h_x(vector<float> x_k) {
		assert(x_k.size() == 6);
		vector<float> result(3);
		result[0]=para_.fx*x_k[0]/ x_k[4] + para_.tx;
		result[1]=para_.fy*x_k[2]/ x_k[4] + para_.ty;
		result[2]=distance / x_k[4];
		return result;
	}
	void F_k(vector<float> u_k, Mat &ans){
		float ans_[6][6] = { 0 };
		ans_[0][0] = cos(para_.ts*u_k[3]);
		ans_[0][1] = para_.ts*cos(para_.ts*u_k[3]);
		ans_[0][4] = -sin(para_.ts*u_k[3]);
		ans_[0][5] = -para_.ts*sin(para_.ts*u_k[3]);
		ans_[1][0] = u_k[3] * cos(para_.ts*u_k[3]);
		ans_[1][1] = (para_.ts*u_k[3] + 1)*cos(para_.ts*u_k[3]);
		ans_[1][4] = -u_k[3] * sin(para_.ts*u_k[3]);
		ans_[1][5] = -(para_.ts*u_k[3] + 1)*sin(para_.ts*u_k[3]);

		ans_[2][2] = 1;
		ans_[2][3] = para_.ts;
		ans_[3][2] = 0;
		ans_[3][3] = 1;

		ans_[4][0] = sin(para_.ts*u_k[3]);
		ans_[4][1] = para_.ts*sin(para_.ts*u_k[3]);
		ans_[4][4] = cos(para_.ts*u_k[3]);
		ans_[4][5] = para_.ts*cos(para_.ts*u_k[3]);
		ans_[5][0] = u_k[3] * sin(para_.ts*u_k[3]);
		ans_[5][1] = (para_.ts*u_k[3] + 1)*sin(para_.ts*u_k[3]);
		ans_[5][4] = u_k[3] * cos(para_.ts*u_k[3]);
		ans_[5][5] = (para_.ts*u_k[3] + 1)*cos(para_.ts*u_k[3]);
		Mat(6, 6, CV_32FC1, ans_).copyTo(ans);
		return;
	}
	void H_k(vector<float> x_k, Mat &ans){
		float ans_[3][6] = { 0 };
		ans_[0][0] = para_.fx / x_k[4];
		ans_[0][4] = -(para_.fx*x_k[0]) / (x_k[4] * x_k[4]);
		ans_[1][2] = para_.fy / x_k[4];
		ans_[1][4] = -(para_.fy*x_k[2]) / (x_k[4] * x_k[4]);
		ans_[2][4] = -distance / (x_k[4] * x_k[4]);
		Mat(3, 6, CV_32FC1, ans_).copyTo(ans);
		return;
	}
	vector<float> vecMinus(vector<float>a, vector<float>b) {
		assert(a.size() == b.size());
		vector<float> ans;
		for (int i = 0; i < a.size(); i++) {
			ans.push_back(a[i] - b[i]);
		}
		return ans;
	}
	vector<float> vecPlus(vector<float>a, vector<float>b) {
		
		assert(a.size() == b.size());
		vector<float> ans;
		for (int i = 0; i < a.size(); i++) {
			ans.push_back(a[i] + b[i]);
		}
		return ans;
	}
	vector<float> toVec(Mat a) {
		assert(a.checkVector(0));
		vector<float> V;
		V.assign((float*)a.datastart, (float*)a.dataend);
		return V;
	}
public:
	void init(vector<float> current_measure, bool reset) {
		pos.resize(3);
		status.resize(6);
		distance=2.0;
		{
			para_.fx = 520;
			para_.fy = 520;
			para_.tx = 320;
			para_.ty = 240;
			para_.ts = 0.05;
			para_.qk = 0.1;
			para_.rk = 0.1;
		}
		rk=Mat::eye(3, 3, CV_32FC1)*para_.rk;
		qk=Mat::eye(6, 6, CV_32FC1)*para_.qk;
		pos[2] = distance / current_measure[2];
		pos[0] = (current_measure[0] - para_.tx) * pos[2] / para_.fx;
		pos[1] = (current_measure[1] - para_.ty) * pos[2] / para_.fy;
		status[0] = pos[0];
		status[1] = 0;
		status[2] = pos[1];
		status[3] = 0;
		status[4] = pos[2];
		status[5] = 0;
		pk = Mat(6, 6, CV_32FC1, Scalar(0.5));
	}
	void update(vector<float> current_measure, float ux, float uy, float uz, float uyaw, long long det_s) {
		Mat hk, fk, sk1, sk2, temp_p1, temp_p2, kk;
		vector<float> diff, temp_status, uk;
		if(det_s) para_.ts = (1.0 * det_s) / 1000000;
		uk.push_back(ux);
		uk.push_back(uy);
		uk.push_back(uz);
		uk.push_back(uyaw);
		temp_status = f_x(uk);
		F_k(uk,fk);
		temp_p1 = fk*pk;
		gemm(temp_p1, fk, 1, qk, 1, temp_p2, GEMM_2_T);
		H_k(temp_status,hk);	
		diff = vecMinus(current_measure, h_x(temp_status));
		sk1 = hk*temp_p2;
		gemm(sk1, hk, 1, rk, 1, sk2, GEMM_2_T);
		solve(sk2, sk1, kk, DECOMP_SVD);
		status = vecPlus(temp_status, toVec(kk.t()*Mat(diff)));
		pos[0] = status[0];
		pos[1] = status[2];
		pos[2] = status[4];
		pk = (Mat::eye(kk.cols, hk.cols, CV_32FC1) - kk.t()*hk)*temp_p2;
	}
	vector<float> position() {
		return pos;
	}
};
