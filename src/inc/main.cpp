
#include <opencv2/highgui.hpp>
#include <iostream>
#include <ctime>
#include <thread>
#include <mutex>
#include"tracker.cpp"

using namespace std;
using namespace cv;

mutex mtxCam;

static Mat image;
static vector<int> boundingBox(4);
static bool paused;
static bool selectObject = false;
static bool startSelection = false;
static int frame_num=0;

void getFrame(VideoCapture *cap, Mat *frame) 
{
	while (true) 
	{
		mtxCam.lock();
		*cap >> *frame;
		mtxCam.unlock();
		frame_num++;
		waitKey(1);
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
			paused = false;
			selectObject = true;
			cout<<"get rect"<<endl;
			break;
		case EVENT_MOUSEMOVE:

			if (startSelection && !selectObject)
			{
				//draw the bounding box
				Mat currentFrame;
				image.copyTo(currentFrame);
				rectangle(currentFrame, Point(boundingBox[0], boundingBox[1]), Point(x, y), Scalar(255, 0, 0), 2, 1);
				imshow("Tracking API", currentFrame);
			}
			break;
		}
	}
}

int main(int argc, char** argv)
{
	//open the capture
	VideoCapture cap;
	cap.open(0);
	if (!cap.isOpened())
	{
		cout << "***Could not initialize capturing...***\n";
		return -1;
	}
	StapleTracker *tracker=new StapleTracker();
	Mat frame;
	paused = true;
	namedWindow("Tracking API", 1);
	setMouseCallback("Tracking API", onMouse, 0);
	cap >> frame;
	frame.copyTo(image);
	imshow("Tracking API", image);
	thread thread_video(getFrame, &cap, &frame);
	cout<<"created thread"<<endl;
	bool initialized = false;
	time_t time0;
	//float fps;
	int i=0;
	VideoWriter writer;  
	int codec = CV_FOURCC('M', 'J', 'P', 'G');  
	double fps = 9.0;
	string filename = "live.avi";
	writer.open(filename, codec, fps, frame.size(), true);
	if (!writer.isOpened()) {  
		cerr << "Could not open the output video file for write\n";  
		return -1;  
	} 
	for (;;)
	{
		if (!paused)
		{
			//mtxCam.lock();
			frame.copyTo(image);
			//mtxCam.unlock();

			if (!initialized && selectObject)
			{
				//initializes the tracker
				if (!tracker->Init(image, boundingBox))
				{
					cout << "***Could not initialize tracker...***\n";
					return -1;
				}
				initialized = true;
				time0=clock();
			}
			else if (initialized)
			{
				//updates the tracker
				i++;
				time_t time1, time2;
				time1 = time2 = clock();
				tracker->runTracker(image);
				//cout << "Main_tracking time:" << clock() - time1 << endl;
				time1 = clock();
				tracker->updateTracker(image);
				//cout << "Main_updating time:" << clock() - time1 << endl;
				time1 = clock();
				tracker->drawCurrentImg(image);
				//cout << "Main_drawing time:" << clock() - time1 << endl;
				cout << "Main_total time:" << (clock() - time2)/(float)CLOCKS_PER_SEC << endl;
				cout << "Total tracking time:" << (clock() - time0)/(float)CLOCKS_PER_SEC << endl;
				//cout << "frame number:" << frame_num << endl;
				fps=((float)CLOCKS_PER_SEC*i)/(clock() - time0);
				cout << "fps:" << fps << endl;
				writer.write(image);
			}
			imshow("Tracking API", image);
		}

		char c = (char)waitKey(2);
		if (c == 'q')
			break;
		if (c == 'p')
			paused = !paused;

	}

	return 0;
}
