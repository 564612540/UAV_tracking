#include <dji_sdk/dji_drone.h>
#include<iostream>
#include<fstream>
#include<ctime>
#include<vector>
#include<queue>
#include<thread>
#include<cassert>
#include<opencv2/core/core.hpp>
//#include <opencv2/opencv.hpp>
#include<ros/ros.h>
#include<thread>
#include<mutex>
#include <sys/time.h>

//#include"inc2/trackOnUav.h"
#include"inc/tracker.cpp"
#include"detector.h"
#include"PID.h"
#include"Kalman.h"
#include"trajectory.h"

#define CONTROL

using namespace std;
using namespace ros;
using namespace cv;
using namespace DJI::onboardSDK;

mutex mtxCam;

void getFrame(VideoCapture *cap, Mat *frame) 
{
    while (true)
    {
        mtxCam.lock();
        *cap >> *frame;
        mtxCam.unlock();
        waitKey(1);
    }
}

long long getSystemTime(){  
    struct timeval tv;  
    gettimeofday(&tv,NULL); 
    return tv.tv_sec*1000000 + tv.tv_usec;  
} 

int main(int argc, char **argv) {

    init(argc, argv, "main");
    NodeHandle nh;
    DJIDrone *drone = new DJIDrone(nh);
    Detector target_finder(argc, argv);
    StapleTracker visual_tracker;
    Kalman motion_estimator;
    Trajectory traj_generator;
    PID controller;
    //LEDcontroller led;
    const float con_gain=0.3;
    queue<float> q_vx, q_vy, q_vz, q_vyaw;
    fstream log_file("log_date.log",ios::trunc|ios::out);
    fstream log_pos("log_pos.csv",ios::trunc|ios::out);
    clock_t timer, timer_fps;
    long long timer_ts;
    char name[64];
    timer=clock();
    int iterate_times=0;
    bool target_lost=false;

    vector<int> bounding_box(4);
    Mat image;
    VideoCapture cap;
    Mat frame;
    cap.open(0);
    if(!cap.isOpened()){
        log_file<<(clock() - timer)/(float)CLOCKS_PER_SEC<<" cannot open camera!"<<endl;
        return 0;
    }else{
        log_file<<(clock() - timer)/(float)CLOCKS_PER_SEC<<" camera opened"<<endl;
    }
    thread thread_video(getFrame, &cap, &frame);

    //first frame init
    while(1){
        frame.copyTo(image);
        while(!target_finder.find(image))
            frame.copyTo(image);
        log_file<<(clock() - timer)/(float)CLOCKS_PER_SEC<<" target found"<<endl;
        //led.on();
        log_file<<(clock() - timer)/(float)CLOCKS_PER_SEC<<" start tracking init"<<endl;
	break;
    }
	//init tracker
    visual_tracker.init(image,target_finder.bounding_box());
	log_file << (clock() - timer) / (float)CLOCKS_PER_SEC << " init tracker" << endl;
	log_pos<<visual_tracker.position()[0] <<"," <<visual_tracker.position()[1] <<"," <<visual_tracker.position()[2] <<",";
	//init EKF
    motion_estimator.init(visual_tracker.position(),true);
	log_file << (clock() - timer) / (float)CLOCKS_PER_SEC << " init kalman" << endl;
	log_pos<<motion_estimator.position()[0] <<"," <<motion_estimator.position()[1] <<"," <<motion_estimator.position()[2] <<",";
	//init trajectory
    traj_generator.init(motion_estimator.position());
	log_pos<<traj_generator.position()[0] <<"," <<traj_generator.position()[1] <<"," <<traj_generator.position()[2] <<"," <<traj_generator.position()[2] <<endl;
	log_file << (clock() - timer) / (float)CLOCKS_PER_SEC << " init traj_gen" << endl;
#ifdef CONTROL
	//init PID
    controller.init("para.txt");
	log_file << (clock() - timer) / (float)CLOCKS_PER_SEC << " start main tracking loop" << endl;
	log_file << (clock() - timer) / (float)CLOCKS_PER_SEC << " access to rc data " << drone->rc_channels.mode << drone->rc_channels.gear <<endl;
	//init UAV control permission
    while(!drone->request_sdk_permission_control());
	log_file << (clock() - timer) / (float)CLOCKS_PER_SEC << " request UAV control" << endl;
	//drone->takeoff();
#endif
    
    timer_ts=getSystemTime();
    timer_fps=clock();
    while(ros::ok()){	//main loop
        frame.copyTo(image);
	ros::spinOnce();
	target_lost=!visual_tracker.track(image);
	if(target_lost){
		log_file << (clock() - timer) / (float)CLOCKS_PER_SEC << " target lost!" << endl;
	}
/*
	if(!visual_tracker.track(image)){
            //led.off();
            log_file << (clock() - timer) / (float)CLOCKS_PER_SEC << " target lost" << endl;
            while(!target_finder.find(image)){
                frame.copyTo(image);
                drone->attitude_control(0x48,0,0,0,0);
            }
            log_file<<(clock() - timer)/(float)CLOCKS_PER_SEC<<" target found"<<endl;
            //led.on();
            visual_tracker.init(image,target_finder.bounding_box());
            motion_estimator.init(visual_tracker.position(),true);
        }
*/
        if(!(iterate_times%5)/*&&!target_lost*/)
		visual_tracker.update(image);
	visual_tracker.drawCurrentImg(image);
	//imshow("Tracking API", image);
	log_pos<<visual_tracker.position()[0] <<"," <<visual_tracker.position()[1] <<"," <<visual_tracker.position()[2] <<",";
	log_pos<<drone->velocity.vx <<"," <<drone->velocity.vy <<"," <<drone->velocity.vz <<",";
        //montion estimator
#ifdef CONTROL
	if(drone->rc_channels.gear==-4545&&q_vx.size()>7){
	    motion_estimator.update(visual_tracker.position(), q_vx.front()*con_gain, q_vy.front()*con_gain, q_vz.front()*con_gain, q_vyaw.front() *3.14159 /180.0 *con_gain, getSystemTime()-timer_ts);
	    q_vx.pop();
	    q_vy.pop();
	    q_vz.pop();
	    q_vyaw.pop();
	}
	else
#endif
	    motion_estimator.update(visual_tracker.position(), 0, 0, 0, 0, getSystemTime()-timer_ts);
	log_pos<<motion_estimator.position()[0] <<"," <<motion_estimator.position()[1] <<"," <<motion_estimator.position()[2] <<",";
	timer_ts=getSystemTime();
        //trajectory generator
        traj_generator.generate(motion_estimator.position());
	log_pos<<traj_generator.position()[0] <<"," <<traj_generator.position()[1] <<"," <<traj_generator.position()[2] <<"," <<traj_generator.position()[3];
#ifdef CONTROL
        //PID controller and UAV communicator
	if(drone->rc_channels.gear==-4545)
	{
        	controller.update(traj_generator.position());
		log_pos<<","<<controller.vx <<"," <<controller.vy <<"," <<controller.vz <<"," <<controller.vyaw <<endl;
        	q_vx.push(controller.vx);
		q_vy.push(controller.vy);
		q_vz.push(controller.vz);
		q_vyaw.push(controller.vyaw);
		drone->attitude_control(0x4A, controller.vx, controller.vy, controller.vz, controller.vyaw);
	}
	else
	{    
		log_pos<<endl;
		controller.init("para.txt");
		
		drone->attitude_control(0x4A, 0, 0, 0, 0);
	}
#endif
#ifndef CONTROL
	log_pos<<endl;
#endif
	if(!(iterate_times%50)){
		vector<int> param = vector<int>(2);
		sprintf(name,"%03d.jpg",iterate_times);  
		param[0]=CV_IMWRITE_JPEG_QUALITY;  
		param[1]=95;  
		imwrite(name,image,param);
	}        
	//write log before & after every step
	iterate_times++;
        log_file << (clock() - timer) / (float)CLOCKS_PER_SEC << " All right, " << (float)(CLOCKS_PER_SEC * iterate_times) / (clock() - timer_fps) << "fps" << endl;
	//waitKey();
    }
#ifdef CONTROL  
    drone->release_sdk_permission_control();
#endif
	log_file << (clock() - timer) / (float)CLOCKS_PER_SEC << " end tracking" << endl;
    return 0;
}
