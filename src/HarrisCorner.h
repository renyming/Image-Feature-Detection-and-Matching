#pragma once

#include<opencv2/opencv.hpp>
#include<vector>
#include "ANMS.h"

void harrisCorner(const cv::Mat&, std::vector<cv::KeyPoint>&, int, bool);
void localMaxima(const cv::Mat& cNorm, std::vector<cv::KeyPoint>& corners, int threshold);