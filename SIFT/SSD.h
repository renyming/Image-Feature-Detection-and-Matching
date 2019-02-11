#pragma once
#include<opencv2/opencv.hpp>
#include"Descriptor.h"
#include<limits>

using namespace std;


void findMatch(std::vector<Descriptor>& keyPointDescriptor1, std::vector<Descriptor>& keyPointDescriptor2, std::vector<cv::KeyPoint>& keyPoint1, std::vector<cv::KeyPoint>& keyPoint2, std::vector<cv::DMatch>& dMatch);
float getDistance(cv::Mat& v1, cv::Mat& v2);
float getFeatureDistance(vector<float> f1, vector<float> f2);