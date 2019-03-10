#pragma once

#include <opencv2/opencv.hpp>
class Descriptor {
public:

	int x;
	int y;
	bool isPaired;
	cv::Mat v;

	Descriptor() {}
	Descriptor(int x, int y, const cv::Mat& v, bool isPaired = false):x(x),y(y),isPaired(isPaired),v(v){}
};