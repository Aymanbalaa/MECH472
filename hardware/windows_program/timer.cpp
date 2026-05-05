
// copied from lectures_examples_prof/bluetooth_robot_1.1_vision/windows_program/timer.cpp

#include <iostream>
#include <cmath>

#include <windows.h>

#include "timer.h"

using namespace std;

double high_resolution_time()
{
	static int init=0;
	static double pow32, count_low0, count_high0, timer_frequency;
	double t, count_low, count_high;
	LARGE_INTEGER count;

	if(init==0) {
		pow32 = pow(2.0,32);

		QueryPerformanceCounter(&count);
		count_low0  = count.LowPart;
		count_high0 = count.HighPart;

		QueryPerformanceFrequency(&count);
		timer_frequency = count.LowPart;

		init=1;
	}

	QueryPerformanceCounter(&count);
	count_low  = count.LowPart  - count_low0;
	count_high = count.HighPart - count_high0;

	t = (count_low + count_high*pow32) / timer_frequency;

	return t;
}


unsigned int high_resolution_count()
{
	LARGE_INTEGER count;
	unsigned int ans;

	QueryPerformanceCounter(&count);

	ans = count.LowPart;

	return ans;
}
