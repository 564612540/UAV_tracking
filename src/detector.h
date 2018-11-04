#include<iostream>
#include<ros/ros.h>
#include<vector>
#include<image_transport/image_transport.h>
#include<opencv2/core/core.hpp>
#include<opencv2/highgui/highgui.hpp>
#include<cv_bridge/cv_bridge.h>

#include"uav_tracking/boundingbox.h"

//using namespace ros;
using namespace std;

class Detector{
private:
	static vector<int> bbox;
	ros::NodeHandle nh;
	static bool bbox_get;
	sensor_msgs::ImagePtr msg;
	image_transport::Publisher pub;
	ros::Subscriber sub;
	static void bboxCallback(uav_tracking::boundingbox b_box){
		bbox[0]=b_box.x;
		bbox[1]=b_box.y;
		bbox[2]=b_box.w;
		bbox[3]=b_box.h;
		bbox_get=true;
		//cout<<"bbox get"<<endl;
	}

public:
	Detector(int argc, char **argv){
		ros::init(argc, argv, "detector");
		image_transport::ImageTransport imtr(nh);
		pub = imtr.advertise("camera/image", 1);
		sub = nh.subscribe("boundingbox", 1, bboxCallback);
		bbox_get=false;
		bbox.resize(4);
	}
	bool find(cv::Mat image){
		msg=cv_bridge::CvImage(std_msgs::Header(), "bgr8", image).toImageMsg();
    		pub.publish(msg);
		//cout<<"publish img"<<endl;
    		ros::spinOnce();
		if(bbox_get){
			bbox_get=false;
			return true;
		}
		return false;
	}
	vector<int> bounding_box(){
		return bbox;
	}
};
vector<int> Detector::bbox={0,0,0,0};
bool Detector::bbox_get=false;
