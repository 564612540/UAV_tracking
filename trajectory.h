#pragma once
#include<vector>
#include<cassert>
//#define SWITCH
using namespace std;
class Trajectory {
private:
	float y, z;
	vector<float> pos;
public:
	void init(vector<float> current_pose) {
		assert(current_pose.size() == 3);
		pos.resize(4);
		y = current_pose[1];
		z = current_pose[2];
	}
	void generate(vector<float> current_pose) {
		assert(current_pose.size() == 3);
#ifdef SWITCH
		if (current_pose[2] < z*0.8||current_pose[2]> z*1.3) {
			pos[0] = 1 - z / current_pose[2];
			pos[1] = (current_pose[0]) / current_pose[2];
			pos[2] = -(current_pose[1] - y) / current_pose[2];
			pos[3] = 0;
		}
		else 
#endif
		{
#ifdef SWITCH
			pos[0] = 0;
#endif
#ifndef SWITCH
			pos[0] = 1 - z / current_pose[2];
#endif		
			pos[1] = 0;
			pos[2] = -(current_pose[1] - y) / current_pose[2];
			pos[3] = (current_pose[0]*z) / current_pose[2];
		}
	}
	vector<float> position() {
		return pos;
	}
};
