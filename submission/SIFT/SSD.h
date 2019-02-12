#pragma once
#include<opencv2/opencv.hpp>
#include"Descriptor.h"
#include<limits>

using namespace std;


void findMatch(std::vector<Descriptor>& keyPointDescriptor1, std::vector<Descriptor>& keyPointDescriptor2, std::vector<cv::KeyPoint>& keyPoint1, std::vector<cv::KeyPoint>& keyPoint2, std::vector<cv::DMatch>& dMatch, int=5);
float getDistance(cv::Mat& v1, cv::Mat& v2);