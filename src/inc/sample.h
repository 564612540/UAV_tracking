#pragma once
#include<vector>
#include<iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
//#include <opencv2/nonfree/nonfree.hpp>
#include <opencv2/features2d/features2d.hpp>
using namespace std;
using namespace cv;
class Sample {
private:
	Mat frame_;
	vector<int> bounding_box_;
	bool frame_is_empty_, bounding_box_is_empty_;
public:
	Sample() {
		frame_is_empty_ = true;
		bounding_box_is_empty_ = true;
	}
	Sample(Mat frame, vector<int> bounding_box) {
		if (bounding_box.size() == 4) {
			frame_ = frame.clone();
			bounding_box_ = bounding_box;
			frame_is_empty_ = false;
			bounding_box_is_empty_ = false;
		}
		else {
			throw "error bounding box parameter!";
		}
	}
	Mat frame() {
		if (frame_is_empty_) {
			throw "frame is empty";
		}
		return frame_;
	}
	vector<int> bounding_box() {
		if (bounding_box_is_empty_) {
			throw "bounding box is empty";
		}
		return bounding_box_;
	}
	void SetFrame(Mat frame) {
		frame_ = frame.clone();
		frame_is_empty_ = false;
	}
	void SetBoundingBox(vector<int> bounding_box) {
		if (bounding_box.size() == 4) {
			bounding_box_ = bounding_box;
			bounding_box_is_empty_ = false;
		}
		else {
			throw "error bounding box parameter!";
		}
	}
	bool IsEmpty() {
		return(!bounding_box_is_empty_&&!frame_is_empty_);
	}
};
