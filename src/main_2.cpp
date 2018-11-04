#include <dji_sdk/dji_drone.h>
#include<iostream>
#include<fstream>
#include<ctime>
#include<vector>
#include<thread>
#include<cassert>
#include<opencv2/core/core.hpp>
//#include <opencv2/opencv.hpp>
#include<ros/ros.h>
#include<thread>
#include<mutex>
#include <sys/types.h>
#include <sys/shm.h>
#include <sys/sem.h>

#include"detector.h"
#include"PID.h"
#include"Kalman.h"
#include"trajectory.h"

using namespace std;
using namespace ros;
using namespace cv;
using namespace DJI::onboardSDK;

mutex mtxCam;

struct transData{
	bool start;
	bool isLost;
	float posi[3];
	int bounding_box[4];
	uchar imageData[921600];
};

void getFrame(VideoCapture *cap, Mat *frame) 
{
    while (true)
    {
        mtxCam.lock();
        *cap >> *frame;
        mtxCam.unlock();
	//imshow("pos", *frame);
                //frame_num++;
        waitKey(1);
    }
}

union semun  
{  
    int val;  
    struct semid_ds *buf;  
    unsigned short *arry;  
};

static int sem_id;

static int set_semvalue();  
static void del_semvalue();  
static int semaphore_p(int num);  
static int semaphore_v(int num);

int main(int argc, char **argv) {
    init(argc, argv, "main");
    NodeHandle nh;
    DJIDrone *drone = new DJIDrone(nh);
    Detector target_finder(argc, argv);
    Kalman motion_estimator;
    Trajectory traj_generator;
    PID controller;
    fstream log_file("log_date.log",ios::trunc|ios::out);
    fstream log_pos("log_pos.csv",ios::trunc|ios::out);
    clock_t timer;
    timer=clock();
    //LEDcontroller led;
    sem_id = semget((key_t)1234, 0, IPC_CREAT);
    if(sem_id == -1)
	{
		perror("semget failed");
		return -1;
	}
    transData *trans;
    key_t mKey=ftok("/key", 1);
    int flag=shmget((key_t) 12345, sizeof(transData), IPC_CREAT);
	if(flag == -1)
	{
		perror("shmget");
		return -1;
	}
	trans = (transData*)shmat(flag,0,0);
	cout<<"trans addr"<<trans<<endl;
	if(trans == (void*) -1)
	{
		perror("shmat failed");
		return -1;
	}
    	bool find=0;
	vector<int> box;
	Mat share_frame(480,640,CV_8UC3,trans->imageData);
    	while(!find)
	{	
		semaphore_p(1);
		find = target_finder.find(share_frame);
		if(find)
		{
		box = target_finder.bounding_box();
    		trans->bounding_box[0] = box[0];
		trans->bounding_box[1] = box[1];
		trans->bounding_box[2] = box[2];
		trans->bounding_box[3] = box[3];
    		trans->start=1;
		}
		semaphore_v(1);
	}
	vector<float> trackerPos(3);
	bool isLost = false;
	
	trackerPos[0] = box[0] + box[2]/2.0;
	trackerPos[1] = box[1] + box[3]/2.0;
	trackerPos[2] = 1;	
	log_pos<<trackerPos[0]<<","<<trackerPos[1]<<","<<trackerPos[2]<<",";
	motion_estimator.init(trackerPos, true);
	log_file << (timer - clock()) / (float)CLOCKS_PER_SEC << " init kalman" << endl;
	log_pos<<motion_estimator.position()[0] <<"," <<motion_estimator.position()[1] <<"," <<motion_estimator.position()[2] <<",";
	//init trajectory
    	traj_generator.init(motion_estimator.position());
	log_pos<<traj_generator.position()[0] <<"," <<traj_generator.position()[1] <<"," <<traj_generator.position()[2] <<"," <<traj_generator.position()[2] <<endl;
	log_file << (timer - clock()) / (float)CLOCKS_PER_SEC << " init traj_gen" << endl;
	//init pid
    	controller.init("para.txt");
	
    	log_file << (timer - clock()) / (float)CLOCKS_PER_SEC << " start main tracking loop" << endl;    	
	
	while(!drone->request_sdk_permission_control());
	log_file << (clock() - timer) / (float)CLOCKS_PER_SEC << " request UAV control" << endl;
	int count = 0;
	while(count<1000)	
	{
	ros::spinOnce();
	semaphore_p(0);
	
	semaphore_p(1);
	isLost = trans->isLost;
	trackerPos[0] = trans->posi[0];
	trackerPos[1] = trans->posi[1];
	trackerPos[2] = trans->posi[2];
	semaphore_v(1);
	log_pos<<trackerPos[0]<<","<<trackerPos[1]<<","<<trackerPos[2]<<",";
	motion_estimator.update(trackerPos,/*controller.vx,controller.vy,controller.vz,controller.vyaw*/0,0,0,0);
	
	log_pos<<motion_estimator.position()[0] <<"," <<motion_estimator.position()[1] <<"," <<motion_estimator.position()[2] <<",";
	
        //trajectory generator
        traj_generator.generate(motion_estimator.position());
	
	log_pos<<traj_generator.position()[0] <<"," <<traj_generator.position()[1] <<"," <<traj_generator.position()[2] <<"," <<traj_generator.position()[2];
	
        if(drone->rc_channels.gear==-4545)
	{
		//drone->request_sdk_permission_control();
        	controller.update(traj_generator.position());
		log_pos<<","<<controller.vx <<"," <<controller.vy <<"," <<controller.vz <<"," <<controller.vyaw <<endl;
        
		drone->attitude_control(0x4A, controller.vx, controller.vy, controller.vz, controller.vyaw);
	}
	else
	{    
		//drone->release_sdk_permission_control();
		log_pos<<endl;
		controller.init("para.txt");
		drone->attitude_control(0x4A, 0, 0, 0, 0);
	}
	count++;
	}
	cout<<"terminated"<<endl;
}


static int set_semvalue()  
{  
    //用于初始化信号量，在使用信号量前必须这样做  
    union semun sem_union;  
  
    sem_union.val = 1;  
    if(semctl(sem_id, 0, SETVAL, sem_union) == -1)  
      {	  perror("semaphore 0 set failed\n");	
	  return 0; }
    if(semctl(sem_id, 1, SETVAL, sem_union) == -1)  
      {	  perror("semaphore 1 set failed\n");	
	  return 0; }
    return 1;  
}  
  
static void del_semvalue()  
{  
    //删除信号量  
    union semun sem_union;  
  
    if(semctl(sem_id, 0, IPC_RMID, sem_union) == -1)  
        perror("semaphore 0 failed to delete\n"); 
    if(semctl(sem_id, 1, IPC_RMID, sem_union) == -1)  
        perror("semaphore 1 failed to delete\n"); 
}  

static int semaphore_p(int num)  
{  
    //对信号量做减1操作，即等待P（sv）  
    struct sembuf sem_b;  
    sem_b.sem_num = num;  
    sem_b.sem_op = -1;//P()  
    sem_b.sem_flg = SEM_UNDO;  
    if(semop(sem_id, &sem_b, 1) == -1)  
    {  
	printf("%d", num);
	perror("semaphore_p failed\n");  
        return 0;  
    }  
    return 1;  
}  
  
static int semaphore_v(int num)  
{  
    //这是一个释放操作，它使信号量变为可用，即发送信号V（sv）  
    struct sembuf sem_b;  
    sem_b.sem_num = num;  
    sem_b.sem_op = 1;//V()  
    sem_b.sem_flg = SEM_UNDO;  
    if(semop(sem_id, &sem_b, 1) == -1)  
    {  
	printf("%d", num);
        perror("semaphore_v failed\n");
        return 0;  
    }  
}
