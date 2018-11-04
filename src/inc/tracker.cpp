#include"tracker.h"

Mat StapleTracker::gaussian_shaped_labels(double sigma, int dim1, int dim2)
{
	cv::Mat labels(dim2, dim1, CV_32FC1);
	int range_y[2] = { -dim2 / 2, dim2 - dim2 / 2 };
	int range_x[2] = { -dim1 / 2, dim1 - dim1 / 2 };

	double sigma_s = sigma*sigma;
	for (int y = range_y[0], j = 0; y < range_y[1]; ++y, ++j) {
		float * row_ptr = labels.ptr<float>(j);
		double y_s = y*y;
		for (int x = range_x[0], i = 0; x < range_x[1]; ++x, ++i) {
			row_ptr[i] = std::exp(-0.5 * (y_s + x*x) / sigma_s);
		}
	}
	return labels;
}
Mat StapleTracker::circshift(const cv::Mat &patch, int x_rot, int y_rot)
{
	cv::Mat rot_patch(patch.size(), CV_32FC1);
	cv::Mat tmp_x_rot(patch.size(), CV_32FC1);

	if (x_rot < 0) {
		cv::Range orig_range(-x_rot, patch.cols);
		cv::Range rot_range(0, patch.cols - (-x_rot));
		patch(cv::Range::all(), orig_range).copyTo(tmp_x_rot(cv::Range::all(), rot_range));

		orig_range = cv::Range(0, -x_rot);
		rot_range = cv::Range(patch.cols - (-x_rot), patch.cols);
		patch(cv::Range::all(), orig_range).copyTo(tmp_x_rot(cv::Range::all(), rot_range));
	}
	else if (x_rot > 0) {
		cv::Range orig_range(0, patch.cols - x_rot);
		cv::Range rot_range(x_rot, patch.cols);
		patch(cv::Range::all(), orig_range).copyTo(tmp_x_rot(cv::Range::all(), rot_range));

		orig_range = cv::Range(patch.cols - x_rot, patch.cols);
		rot_range = cv::Range(0, x_rot);
		patch(cv::Range::all(), orig_range).copyTo(tmp_x_rot(cv::Range::all(), rot_range));
	}
	else {
		cv::Range orig_range(0, patch.cols);
		cv::Range rot_range(0, patch.cols);
		patch(cv::Range::all(), orig_range).copyTo(tmp_x_rot(cv::Range::all(), rot_range));
	}

	if (y_rot < 0) {
		cv::Range orig_range(-y_rot, patch.rows);
		cv::Range rot_range(0, patch.rows - (-y_rot));
		tmp_x_rot(orig_range, cv::Range::all()).copyTo(rot_patch(rot_range, cv::Range::all()));

		orig_range = cv::Range(0, -y_rot);
		rot_range = cv::Range(patch.rows - (-y_rot), patch.rows);
		tmp_x_rot(orig_range, cv::Range::all()).copyTo(rot_patch(rot_range, cv::Range::all()));
	}
	else if (y_rot > 0) {
		cv::Range orig_range(0, patch.rows - y_rot);
		cv::Range rot_range(y_rot, patch.rows);
		tmp_x_rot(orig_range, cv::Range::all()).copyTo(rot_patch(rot_range, cv::Range::all()));

		orig_range = cv::Range(patch.rows - y_rot, patch.rows);
		rot_range = cv::Range(0, y_rot);
		tmp_x_rot(orig_range, cv::Range::all()).copyTo(rot_patch(rot_range, cv::Range::all()));
	}
	else {
		cv::Range orig_range(0, patch.rows);
		cv::Range rot_range(0, patch.rows);
		tmp_x_rot(orig_range, cv::Range::all()).copyTo(rot_patch(rot_range, cv::Range::all()));
	}

	return rot_patch;
}
ComplexMat StapleTracker::gaussian_correlation(const ComplexMat &xf, const ComplexMat &yf, double sigma, bool auto_correlation)
{
	float xf_sqr_norm = xf.sqr_norm();
	float yf_sqr_norm = auto_correlation ? xf_sqr_norm : yf.sqr_norm();

	ComplexMat xyf = auto_correlation ? xf.sqr_mag() : xf * yf.conj();

	cv::Mat xy_sum(xf.rows, xf.cols, CV_32FC1);
	xy_sum.setTo(0);
	cv::Mat ifft2_res = ifft2(xyf);
	for (int y = 0; y < xf.rows; ++y) {
		float * row_ptr = ifft2_res.ptr<float>(y);
		float * row_ptr_sum = xy_sum.ptr<float>(y);
		for (int x = 0; x < xf.cols; ++x) {
			row_ptr_sum[x] = std::accumulate((row_ptr + x*ifft2_res.channels()), (row_ptr + x*ifft2_res.channels() + ifft2_res.channels()), 0.f);
		}
	}

	float numel_xf_inv = 1.f / (xf.cols * xf.rows * xf.n_channels);
	cv::Mat tmp;
	cv::exp(-1.f / (sigma * sigma) * cv::max((xf_sqr_norm + yf_sqr_norm - 2 * xy_sum) * numel_xf_inv, 0), tmp);

	return fft2(tmp);
}
ComplexMat StapleTracker::fft2(const cv::Mat &input)
{
	cv::Mat complex_result(input.size(),CV_32FC2);
	cv::dft(input, complex_result, cv::DFT_COMPLEX_OUTPUT);
	return ComplexMat(complex_result);
}
ComplexMat StapleTracker::fft2(const std::vector<cv::Mat> &input, const cv::Mat &cos_window)
{
	int n_channels = input.size();
	ComplexMat result(input[0].rows, input[0].cols, n_channels); 
	for (int i = 0; i < n_channels; ++i) {
		cv::Mat complex_result;
		cv::dft(input[i].mul(cos_window), complex_result, cv::DFT_COMPLEX_OUTPUT);
		result.set_channel(i, complex_result);
	}
	return result;
}
Mat StapleTracker::ifft2(const ComplexMat &inputf)
{

	cv::Mat real_result;
	if (inputf.n_channels == 1) {
		cv::dft(inputf.to_cv_mat(), real_result, cv::DFT_INVERSE | cv::DFT_REAL_OUTPUT | cv::DFT_SCALE);
	}
	else {
		std::vector<cv::Mat> mat_channels = inputf.to_cv_mat_vector();
		std::vector<cv::Mat> ifft_mats(inputf.n_channels); 
		for (int i = 0; i < inputf.n_channels; ++i) {
			cv::dft(mat_channels[i], ifft_mats[i], cv::DFT_INVERSE | cv::DFT_REAL_OUTPUT | cv::DFT_SCALE);
		}
		cv::merge(ifft_mats, real_result);
	}
	return real_result;
}
Mat StapleTracker::cosine_window_function(int dim1, int dim2)
{
	cv::Mat m1(1, dim1, CV_32FC1), m2(dim2, 1, CV_32FC1);
	double N_inv = 1. / (static_cast<double>(dim1) - 1.);
	for (int i = 0; i < dim1; ++i)
		m1.at<float>(i) = 0.5*(1. - std::cos(2. * CV_PI * static_cast<double>(i) * N_inv));
	N_inv = 1. / (static_cast<double>(dim2) - 1.);
	for (int i = 0; i < dim2; ++i)
		m2.at<float>(i) = 0.5*(1. - std::cos(2. * CV_PI * static_cast<double>(i) * N_inv));
	cv::Mat ret = m2*m1;
	return ret;
}

Mat StapleTracker::getSubWindow(Mat img,Point2i size,Point2i area){
	Mat ROI;
	int x, y, w, h, left = 0, right = 0, top = 0, bottom = 0;
	x = para_.target_center.x - area.x / 2;
	y = para_.target_center.y - area.y / 2;
	w = area.x;
	h = area.y;
	if (x < 0) { left = -x; x = 0; }
	if (y < 0) { top = -y; y = 0; }
	if (x + w>img.cols) {
		right = x + w - img.cols; w = img.cols - x-1;
	}
	if (y + h>img.rows) {
		bottom = y + h - img.rows; h = img.rows - y-1;
	}
	ROI = img(Rect(x, y, w, h));
	Mat temp;
	copyMakeBorder(ROI, temp, top, bottom, left, right, BORDER_REPLICATE);
	Mat out;
	resize(temp, out, (Size)size);
	return out;
}
void StapleTracker::calcHistogram(Mat img,Mat mask,Mat &hist){  
	assert(img.size() == mask.size());
	Mat new_img(img.size(),CV_32FC1,Scalar(0));
	vector<Mat> rgb_img;
	split(img/(256 / para_.n_bins),rgb_img);
	add(rgb_img[0], new_img, new_img, Mat(), CV_32F);
	new_img*=(256 / para_.n_bins);
	add(rgb_img[1], new_img, new_img, Mat(), CV_32F);
	new_img*=(256 / para_.n_bins);
	add(rgb_img[2], new_img, new_img, Mat(), CV_32F);
	int binSize = para_.n_bins*para_.n_bins*para_.n_bins;  
	float range[] = { 0, (float)binSize} ;   
	const float* histRange = { range };  
	bool uniform = true; bool accumulate = false;
	calcHist( &new_img, 1, 0, mask, hist, 1, &binSize, &histRange, uniform, accumulate );
}
vector<Mat> StapleTracker::getFeatureMap(Mat sample){
	Mat gray((Size)para_.cf_size, CV_32F);
	Mat sample_gray;
	cvtColor(sample, sample_gray, CV_RGB2GRAY);
	resize(sample_gray, gray, (Size)para_.cf_size);
	vector<Mat> temp = FHoG::extract(sample_gray, 2, para_.hog_cell_area);
	return temp;
}
Mat StapleTracker::getHistScore(Mat sample){
	Mat pixel_score(sample.size(), CV_32FC1, Scalar(0));
	Mat new_img(sample.size(),CV_32FC1,Scalar(0));
	vector<Mat> rgb_img;
	split(sample/(256 / para_.n_bins),rgb_img);
	add(rgb_img[0], new_img, new_img, Mat(), CV_32F);
	new_img*=(256 / para_.n_bins);
	add(rgb_img[1], new_img, new_img, Mat(), CV_32F);
	new_img*=(256 / para_.n_bins);
	add(rgb_img[2], new_img, new_img, Mat(), CV_32F);
	Mat f(sample.size(), CV_32FC1, Scalar(0)),b(sample.size(), CV_32FC1, Scalar(0));
	int binSize = para_.n_bins*para_.n_bins*para_.n_bins;  
	float range[] = { 0, (float)binSize} ;   
	const float* histRange = { range };  
	calcBackProject(&new_img, 1, 0, hist_fg, f, &histRange);
	calcBackProject(&new_img, 1, 0, hist_bg, b, &histRange);
	pixel_score=f/(f+b);
	Mat out(pixel_score.rows - para_.norm_target_size.y, pixel_score.cols - para_.norm_target_size.x, CV_32FC1, Scalar(0));
	Mat temp_blur;
	boxFilter(pixel_score,temp_blur,-1,(Size)para_.norm_target_size);
	Mat ROI=pixel_score(Rect(para_.norm_target_size.x/2, para_.norm_target_size.y/2, out.cols, out.rows));
	ROI.copyTo(out);
	return out;
}
Mat StapleTracker::getCFScore(Mat sample){
	vector<Mat> feature_map = getFeatureMap(sample);
	Mat out = ifft2(((cf_mats_num / cf_mats_den)*fft2(feature_map, hann_window)).sum_over_channels());
	return out;
}
Mat StapleTracker::mergeScore(Mat hist_score,Mat cf_score){
	Mat temp1,temp2;
	resize(cf_score, temp2, hist_score.size());
	return hist_score*para_.merge_factor + (1 - para_.merge_factor)*temp2;
}
void StapleTracker::updateHist(Mat const img){
	static Mat bg_mask(para_.norm_bg_size.y, para_.norm_bg_size.x, CV_8UC1,Scalar(1));
	static Mat fg_mask(para_.norm_bg_size.y, para_.norm_bg_size.x, CV_8UC1,Scalar(0));
	if (para_.is_first_frame) {
		Rect bg_mask_box=Rect((para_.norm_bg_size.x - para_.norm_target_size.x) / 2,(para_.norm_bg_size.y - para_.norm_target_size.y) / 2,(para_.norm_bg_size.x + para_.norm_target_size.x) / 2,(para_.norm_bg_size.y + para_.norm_target_size.y) / 2);
		Rect fg_mask_box=Rect((para_.norm_bg_size.x - para_.norm_fg_size.x) / 2,(para_.norm_bg_size.y - para_.norm_fg_size.y) / 2,(para_.norm_bg_size.x + para_.norm_fg_size.x) / 2,(para_.norm_bg_size.y + para_.norm_fg_size.y) / 2);
		Mat ROI=bg_mask(bg_mask_box);
		ROI=Scalar(0);
		ROI=fg_mask(fg_mask_box);
		ROI=Scalar(1);
		calcHistogram(img, bg_mask, hist_bg);
		calcHistogram(img, fg_mask, hist_fg);
	}
	else {
		Mat new_hist;
		calcHistogram(img, bg_mask, new_hist);
		hist_bg = (1 - para_.hist_learning_rate)*hist_bg + para_.hist_learning_rate*new_hist;
		calcHistogram(img, fg_mask, new_hist);
		hist_fg = (1 - para_.hist_learning_rate)*hist_fg + para_.hist_learning_rate*new_hist;
	}
}
void StapleTracker::updateCF(Mat sample){
	vector<Mat> feature_map = getFeatureMap(sample);
	if (para_.is_first_frame) {
		hann_window = cosine_window_function(feature_map[0].cols, feature_map[0].rows);
		float output_sigma = sqrt(para_.norm_target_size.x*para_.norm_target_size.y)*para_.output_sigma_factor / para_.hog_cell_area;
		gauss_window = fft2(gaussian_shaped_labels(output_sigma, feature_map[0].cols, feature_map[0].rows));
		feature_mats = fft2(feature_map, hann_window);
		cf_mats_num = feature_mats.conj().mul(gauss_window);
		cf_mats_den = feature_mats*feature_mats.conj();
	}
	else {
		feature_mats = fft2(feature_map, hann_window);
		cf_mats_num = cf_mats_num*(1 - para_.cf_learning_rate) + feature_mats.conj().mul(gauss_window)*para_.cf_learning_rate;
		cf_mats_den = cf_mats_den*(1 - para_.cf_learning_rate) + feature_mats*feature_mats.conj()*para_.cf_learning_rate;
	}
}
