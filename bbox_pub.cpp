#include<iostream>
#include<ros/ros.h>
#include<vector>
#include<image_transport/image_transport.h>
#include<opencv2/core/core.hpp>
//#include <opencv2/opencv.hpp>
#include<opencv2/highgui/highgui.hpp>
#include<cv_bridge/cv_bridge.h>

#include"uav_tracking/boundingbox.h"

using namespace std;
using namespace cv;

Mat image;
static vector<int> boundingBox(4);
static bool selectObject = false;
static bool startSelection = false;

void imageCallback(const sensor_msgs::ImageConstPtr& msg)
{
  try
  {
    image=cv_bridge::toCvShare(msg, "bgr8")->image;
    //cv::imshow("view", image);
    //cout<<"image get"<<endl;
  }
  catch (cv_bridge::Exception& e)
  {
    ROS_ERROR("Could not convert from '%s' to 'bgr8'.", msg->encoding.c_str());
  }
}

static void onMouse(int event, int x, int y, int, void*)
{
	if (!selectObject)
	{
		switch (event)
		{
		case EVENT_LBUTTONDOWN:
			//set origin of the bounding box
			startSelection = true;
			boundingBox[0] = x;
			boundingBox[1] = y;
			break;
		case EVENT_LBUTTONUP:
			//sei with and height of the bounding box
			boundingBox[2] = std::abs(x - boundingBox[0]);
			boundingBox[3] = std::abs(y - boundingBox[1]);
			selectObject = true;
			break;
		case EVENT_MOUSEMOVE:
			if(!startSelection){
				imshow("view", image);
			} else
			if (startSelection && !selectObject)
			{
				//draw the bounding box
				Mat currentFrame;
				image.copyTo(currentFrame);
				rectangle(currentFrame, Point(boundingBox[0], boundingBox[1]), Point(x, y), Scalar(255, 0, 0), 2, 1);
				imshow("view", currentFrame);
			}
			break;
		}
	}
}

int main(int argc, char **argv)
{
  ros::init(argc, argv, "image_listener");
  ros::NodeHandle nh;
  cv::namedWindow("view");
  cv::startWindowThread();
  uav_tracking::boundingbox bbox;
  image_transport::ImageTransport it(nh);
  image_transport::Subscriber sub = it.subscribe("camera/image", 1, imageCallback);
  ros::Publisher pub=nh.advertise<uav_tracking::boundingbox>("boundingbox", 1);
  setMouseCallback("view", onMouse, 0);
  while(ros::ok()){
    if(selectObject){
      bbox.x=boundingBox[0];
      bbox.y=boundingBox[1];
      bbox.w=boundingBox[2];
      bbox.h=boundingBox[3];
      cout<<"publish bbox"<<endl;
      pub.publish(bbox);
      selectObject=false;
      startSelection=false;
      usleep(100000);
	exit(1);
    }
    ros::spinOnce();
  }
  cv::destroyWindow("view");
}
