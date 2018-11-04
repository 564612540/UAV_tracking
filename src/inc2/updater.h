#pragma once
#include <iostream>

#include "parameters.h"

using namespace std;

class ClassificationScoreJudger
{
public:
	inline bool doUpdate(float confidence)
	{
		if (confidence > UPDATE_JUDGER)
			return true;
		return false;
	 }
};