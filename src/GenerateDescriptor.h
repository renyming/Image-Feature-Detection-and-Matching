#pragma once

#define _USE_MATH_DEFINES

#include<opencv2/opencv.hpp>
#include<vector>
#include<cmath>
#include"Descriptor.h"
#include"Key_Point.h"


const float WSIZE = 16;

void generateDescriptor(const cv::Mat& in, const std::vector<cv::KeyPoint>, std::vector<Descriptor>& keyPointDescriptor);
void kPRotation(const cv::Mat& in, const std::vector<cv::KeyPoint>& vKeyPoint, std::vector<cv::KeyPoint>& vKeyPointRotation);
void extractDescriptor(const cv::Mat&, const std::vector<cv::KeyPoint>&, std::vector<Descriptor>&);
void normVector(std::vector<float>&);
void getGaussianWeight(cv::Mat&, int);
cv::Size getKSize(int);
float getSigma(int);