#include"HarrisCorner.h"

using namespace cv;
using namespace std;

void harrisCorner(const Mat& img, vector<KeyPoint>& corners, int threshold, bool isANMS) {
	Mat greyImg;
	cvtColor(img, greyImg, COLOR_BGR2GRAY);
	//GaussianBlur(greyImg, greyImg, Size(3, 3),1);
	Mat Ix, Iy;
	Scharr(greyImg, Ix, CV_32F, 1, 0);
	Scharr(greyImg, Iy, CV_32F, 0, 1);
	Mat Ix2, Iy2, IxIy;
	pow(Ix,2.0,Ix2);
	pow(Iy, 2.0, Iy2);
	multiply(Ix, Iy, IxIy);

	Mat GIx2, GIy2, GIxIy;
	GaussianBlur(Ix2, GIx2, Size(9, 9), 1);
	GaussianBlur(Iy2, GIy2, Size(9, 9), 1);
	GaussianBlur(IxIy, GIxIy, Size(9, 9), 1);

	Mat det, det1, det2;
	multiply(GIx2, GIy2, det1);
	pow(GIxIy, 2.0, det2);
	det = det1 - det2;

	Mat c, cNorm;
	c = det/(GIx2+GIy2);

	normalize(c, cNorm, 0, 255, NORM_MINMAX, CV_32F);
	
	if (isANMS) {
		std::vector<KeyPoint> cornersANMS;
		ANMS(cNorm, cornersANMS, threshold);

		Mat localMax = Mat::zeros(cNorm.size(), CV_32F);
		for (KeyPoint point : cornersANMS) {
			localMax.at<float>(point.pt) = point.response;
		}
		localMaxima(localMax, corners, 0);
	}
	else {
		localMaxima(cNorm, corners, threshold);
	}

}

void localMaxima(const cv::Mat& cNorm, std::vector<KeyPoint>& corners, int threshold) {

	Mat cThreshold = Mat::zeros(cNorm.size(), CV_32F);

	int rows = cNorm.rows;
	int cols = cNorm.cols;
	for (int i = 0; i < rows; ++i) {
		for (int j = 0; j < cols; ++j) {
			float intensity = cNorm.at<float>(i, j);
			if (intensity >= threshold) {
				cThreshold.at<float>(i, j) = intensity;
			}
		}
	}

	for (int i = 1; i < cThreshold.rows - 2; ++i) {
		for (int j = 1; j < cThreshold.cols - 2; ++j) {
			//loop within 3*3 box
			float max = 0;
			int row=-1, col=-1;
			for (int m = -1; m < 2; ++m) {
				for (int n = -1; n < 2; ++n) {
					float current = cThreshold.at<float>(i+m, j+n);
					if (current!=0 && current >= max) {
						max = current;
						//set previous max to 0
						if (row != -1) cThreshold.at<float>(row, col) = 0.0F;
						row = i + m;
						col = j + n;
					}
					else {
						cThreshold.at<float>(i+m,j+n) = 0.0F;
					}
				}
			}

		}
	}

	for (int i = 0; i < rows; ++i) {
		for (int j = 0; j < cols; ++j) {
			float intensity = cThreshold.at<float>(i, j);
			if (intensity != 0) {
				KeyPoint kPoint(Point(j, i), 5, -1.0F, intensity);
				corners.push_back(kPoint);
			}
		}
	}

}
