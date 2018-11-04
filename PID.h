#pragma once
#include<vector>
#include<iostream>
#include<fstream>
#include<assert.h>
using namespace std;
class PID {
private:
	vector<float> last;
	//vector<float> last_diff;
	vector<float> sum;
	vector<float> Kp;
	vector<float> Ki;
	vector<float> Kd;
public:
	float vx, vy, vz, vyaw;
	void init(const char* filename) {
		vx = 0;
		vy = 0;
		vz = 0;
		vyaw = 0;
		last.resize(4);
		last[0] = 0;
		last[1] = 0;
		last[2] = 0;
		last[3] = 0;
		sum.resize(4);
		sum[0] = 0;
		sum[1] = 0;
		sum[2] = 0;
		sum[3] = 0;
		/*last_diff.resize(4);
		last_diff[0] = 0;
		last_diff[1] = 0;
		last_diff[2] = 0;
		last_diff[3] = 0;*/
		ifstream setupfile(filename);
		//if(setupfile.good())cout<<"file open"<<endl;
		//else cout<<"open failed"<<endl;
		Kp.resize(4);
		for (int i = 0; i < 4; i++) {
			setupfile >> Kp[i];
			//cout<<Kp[i]<<endl;
		}
		Ki.resize(4);
		for (int i = 0; i < 4; i++) {
			setupfile >> Ki[i];
		}
		Kd.resize(4);
		for (int i = 0; i < 4; i++) {
			setupfile >> Kd[i];
			//cout<<Kd[i]<<endl;
		}
	}
	void update(vector<float> current) {
		assert(current.size() >= 4);
		//cout<<current[3]<<endl;
		vector<float> diff(4);
		for (int i = 0; i < 4; i++) {
			diff[i] = current[i] - last[i];
			sum[i] += current[i];
			last[i] = current[i];
		}
		vx = Kp[0] * current[0] + Ki[0] * sum[0] + Kd[0] * diff[0];
		vy = Kp[1] * current[1] + Ki[1] * sum[1] + Kd[1] * diff[1];
		vz = Kp[2] * current[2] + Ki[2] * sum[2] + Kd[2] * diff[2];
		vyaw = Kp[3] * current[3] + Ki[3] * sum[3] + Kd[3] * diff[3];
	}
};
