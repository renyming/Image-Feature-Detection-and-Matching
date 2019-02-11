#pragma once

#include <opencv2/opencv.hpp>
class Descriptor {
public:

	int x;
	int y;
	cv::Mat v;

	Descriptor(int x, int y, const cv::Mat& v):x(x),y(y),v(v){}
};