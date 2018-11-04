#include <vector>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <iostream>
#include <iomanip>
#include <numeric>
#include <ctime>
#include <omp.h> 

#include"sample.h"
#include"para.h"
#include"fhog/fhog.hpp"
#include"complexmat.hpp"
//#include"tracker.cpp"

using namespace std;
using namespace cv;

class StapleTracker {
private:
	Para para_;
	float current_scale;
	int lost_count;
	vector<float> scales;
	Mat hann_window,current_sample;
	Mat hist_bg, hist_fg;
	vector<Mat> cf_den, cf_num;
	ComplexMat gauss_window, cf_mats_den, cf_mats_num, feature_mats;
	Mat getSubWindow(Mat img,Point2i size,Point2i area);
	void calcHistogram(Mat img,Mat mask,Mat &hist);
	vector<Mat> getFeatureMap(Mat sample);
	Mat getHistScore(Mat sample);
	Mat getCFScore(Mat sample);
	Mat mergeScore(Mat hist_score,Mat cf_score);
	void updateHist(Mat const img);
	void updateCF(Mat sample);
	cv::Mat gaussian_shaped_labels(double sigma, int dim1, int dim2);
	ComplexMat gaussian_correlation(const ComplexMat & xf, const ComplexMat & yf, double sigma, bool auto_correlation = false);
	cv::Mat circshift(const cv::Mat & patch, int x_rot, int y_rot);
	cv::Mat cosine_window_function(int dim1, int dim2);
	ComplexMat fft2(const cv::Mat & input);
	ComplexMat fft2(const std::vector<cv::Mat> & input, const cv::Mat & cos_window);
	cv::Mat ifft2(const ComplexMat & inputf);
public:
	StapleTracker() {
		para_.hog_cell_area = 4;
		para_.fixed_area = 150 * 150;
		para_.n_bins = 32;
		para_.hist_learning_rate = 0.04;
		para_.inner_padding = 0.2;
		para_.output_sigma_factor = 1.0 / 16.0;
		para_.lambda = 0.001;
		para_.cf_learning_rate = 0.01;
		para_.merge_factor = 0.3;
		para_.den_per_channel = false;
		para_.scale_adaption = true;
		para_.hog_scale_cell_area = 4;
		para_.scale_learning_rate = 0.025;
		para_.scale_sigma_factor = 1.0 / 4.0;
		para_.scale_num = 33;
		para_.scale_model_factor = 1;
		para_.scale_step = 1.01;
		para_.scale_model_max_area = 32 * 16;
		para_.lost_threshold=0.2;
		scales.push_back(1.1);
		scales.push_back(1);
		scales.push_back(1.0 / 1.1);
	}
	bool init(Mat frame,vector<int> bounding_box) {
		if (bounding_box.size() != 4)return false;
		if (frame.channels() == 1)
			para_.is_color = false;
		else 
			para_.is_color = true;
		//cout<<"bounding_box"<<bounding_box[0]<<","<<bounding_box[3]<<endl;
		para_.is_first_frame = true;
		para_.img_size.x = frame.cols;
		para_.img_size.y = frame.rows;
		para_.target_pos.x = bounding_box[0];
		para_.target_pos.y = bounding_box[1];
		para_.target_size.x = bounding_box[2];
		para_.target_size.y = bounding_box[3];
		para_.target_center.x = bounding_box[0] + bounding_box[2] / 2;
		para_.target_center.y = bounding_box[1] + bounding_box[3] / 2;
		para_.bg_size.x = para_.target_size.x*1.5 + para_.target_size.y*0.5;
		para_.bg_size.y = para_.target_size.x*0.5 + para_.target_size.y*1.5;
		if (para_.bg_size.x > para_.img_size.x)para_.bg_size.x = para_.img_size.x - 1;
		if (para_.bg_size.y > para_.img_size.y)para_.bg_size.y = para_.img_size.y - 1;
		para_.fg_size.x = para_.target_size.x*(1 - para_.inner_padding*0.5) - para_.target_size.y*para_.inner_padding*0.5;
		para_.fg_size.y = para_.target_size.y*(1 - para_.inner_padding*0.5) - para_.target_size.x*para_.inner_padding*0.5;
		para_.bg_size.x -= para_.bg_size.x % 2;
		para_.bg_size.y -= para_.bg_size.y % 2;
		para_.fg_size.x += para_.fg_size.x % 2;
		para_.fg_size.y += para_.fg_size.y % 2;
		current_scale = 1;
		para_.area_norm_factor = sqrt(para_.fixed_area*1.0 / (para_.bg_size.x*para_.bg_size.y));
		para_.norm_bg_size = para_.bg_size*para_.area_norm_factor;
		para_.norm_fg_size = para_.fg_size*para_.area_norm_factor;
		para_.norm_bg_size.x -= para_.norm_bg_size.x % 2;
		para_.norm_bg_size.y -= para_.norm_bg_size.y % 2;
		para_.norm_fg_size.x += para_.norm_fg_size.x % 2;
		para_.norm_fg_size.y += para_.norm_fg_size.y % 2;
		para_.cf_size = Point(para_.norm_bg_size.x / para_.hog_cell_area, para_.norm_bg_size.y / para_.hog_cell_area);
		para_.cf_size.x += para_.cf_size.x % 2;
		para_.cf_size.y += para_.cf_size.y % 2;
		para_.norm_target_size.x = para_.norm_bg_size.x*0.75 - para_.norm_bg_size.y*0.25;
		para_.norm_target_size.y = para_.norm_bg_size.y*0.75 - para_.norm_bg_size.x*0.25;
		para_.norm_delta_size.x = para_.norm_delta_size.y = 2 * min((para_.norm_bg_size.x - para_.norm_target_size.x)*0.5, (para_.norm_bg_size.y - para_.norm_target_size.y)*0.5);
		para_.norm_pwp_size = para_.norm_target_size + para_.norm_delta_size;
		frame.copyTo(current_sample);
		Mat patch_padded = getSubWindow(current_sample, para_.norm_bg_size, para_.bg_size);
			//cout<<"init^^^"<<endl;
		updateHist(patch_padded);
		updateCF(patch_padded);
		lost_count=0;
		return true;
	}
	bool track(Mat const frame) {
		frame.copyTo(current_sample);
		if (para_.scale_adaption) {
			vector<double> max_val(scales.size());
			vector<Point2i> max_loc(scales.size());
			Mat score_map;
			for (int i = 0; i < scales.size(); i++) {
				Mat im_patch_cf = getSubWindow(current_sample, para_.norm_bg_size, para_.bg_size*scales[i] * current_scale);
				Mat im_patch_hist = getSubWindow(current_sample, para_.norm_pwp_size, para_.norm_pwp_size*(1.0 / para_.area_norm_factor)*scales[i] * current_scale);
				score_map = mergeScore(getHistScore(im_patch_hist), getCFScore(im_patch_cf));
				minMaxLoc(score_map, NULL, &max_val[i], NULL, &max_loc[i]);
			}
			vector<double>::iterator max_pos = max_element(max_val.begin(), max_val.end());
			int i = distance(max_val.begin(), max_pos);
			if (para_.is_first_frame) para_.init_score=max_val[i];
			else if(max_val[i]<para_.init_score*para_.lost_threshold) lost_count++;
			else lost_count=0;
			current_scale *= scales[i];
			para_.target_center += (max_loc[i] - Point2i(score_map.size())*0.5)*(1.0 / para_.area_norm_factor)*current_scale;
			para_.target_size*=scales[i];
			para_.target_pos = para_.target_center - para_.target_size*0.5;
		}
		if(lost_count>30){
			lost_count=0;
			return false;
		}
		return true;
	}
	bool update(Mat frame) {
		Mat im_patch_padded = getSubWindow(frame, para_.norm_bg_size, para_.bg_size*current_scale);
		updateHist(im_patch_padded);
		updateCF(im_patch_padded);
		if (para_.is_first_frame)para_.is_first_frame = false;
		return true;
	}
	bool drawCurrentImg(Mat image){
		rectangle(image,Rect(para_.target_pos.x,para_.target_pos.y,para_.target_size.x,para_.target_size.y), Scalar(255, 0, 0));
		//cout << para_.target_center << " " << current_scale << endl;
		//imshow("Tracking API", image);
		return true;
	}
	vector<float> position(){
		vector<float> ans;
		ans.resize(3);
		ans[0]=para_.target_center.x;
		ans[1]=para_.target_center.y;
		ans[2]=current_scale;
		return ans;
	}
};
