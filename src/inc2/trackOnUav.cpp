#include "trackOnUav.h"

StapleTracker::StapleTracker()
{
	extractor = HogRawPixelNormExtractor();
	lr =LogisticRegression();
	sampler = SlidingWindowSampler();
	motion = ParticleFilterMotionModel();
	judger = ClassificationScoreJudger();
	worker = new threadpool(4);
	probs = Mat(1, 1, CV_32FC1);
	probs.at<float>(0) = 1.0;
}

StapleTracker::~StapleTracker()
{
	delete worker;
}

void StapleTracker::init(Mat image, Mat rect)
{
	img = img;
	rects = rect;
	maxIdx[1] = 0;
}

bool StapleTracker::init(Mat image, vector<int> bounding_box)
{
	img = image;
	cout<<bounding_box[0]<<bounding_box[3]<<endl;
	rects = Mat(1,4,CV_32FC1);
	rects.at<float>(0)=bounding_box[0];
	rects.at<float>(1)=bounding_box[1];
	rects.at<float>(2)=bounding_box[2];
	rects.at<float>(3)=bounding_box[3];
	cout<<rects<<endl;
	maxIdx[1] = 0;
	maxIdx[0]=0;
	maxProb=1.;
	return true;
}

bool StapleTracker::track(Mat image)
{
	if (lr.isEmpty())
	{
		key_t mKey=ftok("key", 1);
		cout<<mKey<<endl;
		int flag=shmget(mKey, 4, IPC_CREAT);
		shmat(flag,0,0);
		return true;
	}
	img=image;
	
	motion.genParticle(rects, probs);
	extractor.extract(img, rects, feat);
	probs = lr.LR(feat.colRange(0,rects.rows));
	
	minMaxIdx(probs, NULL, &maxProb, NULL, maxIdx);
	if (maxProb > 0.5)
		return true;
	else
		return false;
}

bool StapleTracker::update(Mat image)
{
	
	if (judger.doUpdate(maxProb))
	{	
		
		if (feat.empty())
		{
			posSample = sampler.quickPos(rects);
			negSample = sampler.quickNeg(rects);
			
			posFeature = extractor.extract(img, posSample);
			negFeature = extractor.extract(img, negSample);
			
			lr.trainLR(posFeature, negFeature);
			
			feat = Mat(posFeature.rows, PARTICLE_N, posFeature.type());
			return true;
		}
		else
		{	
			negSample = sampler.quickNeg(rects.row(maxIdx[1]));
			posFeature = extractor.extract(img, posSample);
			
			posSample = sampler.quickPos(rects.row(maxIdx[1]));
			negFeature = extractor.extract(img, negSample);
		
			trainF = worker->commit(trainLR_Thr, &lr, posFeature, negFeature);
		}
	}
	return false;
}

vector<float> StapleTracker::position()
{
	vector<float> res(3);
	const float* ptr =(float*) rects.row(maxIdx[1]).data;
	res[0] = ptr[0]+ptr[2]/2;
	res[1] = ptr[1]+ptr[3]/2;
	res[2] = 1;
	return res;
}

bool StapleTracker::drawCurrentImg(Mat image)
{
	float* tmp = (float*)rects.row(maxIdx[1]).data;
	
	cout<<tmp[0]<<tmp[1]<<tmp[2]<<tmp[3];
	Rect rec=Rect(tmp[0], tmp[1], tmp[2], tmp[3]);
	cout<<rec<<endl;
	rectangle(img, rec,Scalar(255));
	imshow("object",img);
	
	return true;
}

int main()
{	
	
	sem_id = semget((key_t)1234, 1, IPC_CREAT);
	set_semvalue();	
	
	transData *trans;
	key_t mKey=ftok("/key", 1);
	int flag=shmget(mKey, sizeof(transData), 0644);
	trans = (transData*)shmat(flag,0,0);
	trans->start=0;
	
	StapleTracker tracker;
	VideoCapture vc(0);
	vector<int> rect={50,50,100,100};
	cout << rect[0] << endl;
	if (!vc.isOpened())
	{
		return -1;
	}
	Mat frame;
	vc >> frame;
	while(!trans->start)
	{
		semaphore_p();
		//可以再进行判断是否得到bounding box 这样可以使两边的帧保持一致
		frame.copyTo(trans->image);
		semaphore_v();
	}
	imshow("frame1",frame);
	waitKey(1);
	{
	semaphore_p();
	rect[0] = trans->bounding_box[0];
	rect[1] = trans->bounding_box[1];
	rect[2] = trans->bounding_box[2];
	rect[3] = trans->bounding_box[3];
	semaphore_v();
	}
	tracker.init(frame, rect);

	bool isLost=0;
	vector<float> position;
	clock_t start,finish;
	while (1) {
		start=clock();
		isLost = tracker.track(frame);
		tracker.update(frame);
		position = tracker.position();
		{
			semaphore_p();
			trans->isLost = isLost;
			trans->posi[0] = position[0];
			trans->posi[1] = position[1];
			trans->posi[2] = position[2];
			semaphore_v();
		}
		tracker.drawCurrentImg(frame);
		vc>>frame;
		finish=clock();
		cout<<"costed  time is:"<<finish-start<<endl;
	}
}

static int set_semvalue()  
{  
    //用于初始化信号量，在使用信号量前必须这样做  
    union semun sem_union;  
  
    sem_union.val = 1;  
    if(semctl(sem_id, 0, SETVAL, sem_union) == -1)  
        return 0;  
    return 1;  
}  
  
static void del_semvalue()  
{  
    //删除信号量  
    union semun sem_union;  
  
    if(semctl(sem_id, 0, IPC_RMID, sem_union) == -1)  
        fprintf(stderr, "Failed to delete semaphore\n");  
}  
  
static int semaphore_p()  
{  
    //对信号量做减1操作，即等待P（sv）  
    struct sembuf sem_b;  
    sem_b.sem_num = 0;  
    sem_b.sem_op = -1;//P()  
    sem_b.sem_flg = SEM_UNDO;  
    if(semop(sem_id, &sem_b, 1) == -1)  
    {  
        fprintf(stderr, "semaphore_p failed\n");  
        return 0;  
    }  
    return 1;  
}  
  
static int semaphore_v()  
{  
    //这是一个释放操作，它使信号量变为可用，即发送信号V（sv）  
    struct sembuf sem_b;  
    sem_b.sem_num = 0;  
    sem_b.sem_op = 1;//V()  
    sem_b.sem_flg = SEM_UNDO;  
    if(semop(sem_id, &sem_b, 1) == -1)  
    {  
        fprintf(stderr, "semaphore_v failed\n");  
        return 0;  
    }  


