#pragma once

#include<opencv2/opencv.hpp>
#include<vector>

std::vector<cv::KeyPoint> harrisCorner(const cv::Mat&, int);
void localMaxima(cv::Mat&);