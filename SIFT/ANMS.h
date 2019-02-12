#pragma once
#include<opencv2/opencv.hpp>
#include<climits>

void ANMS(const cv::Mat& cNorm, std::vector<cv::KeyPoint>& corners, int threshold);
float getDistance(const cv::KeyPoint& p1, const cv::KeyPoint& p2);
bool compDist(const cv::KeyPoint& p1, const cv::KeyPoint& p2);
void ANMS(std::vector<cv::KeyPoint>& keyPoint, std::vector<cv::KeyPoint>& corners);