#include"HarrisCorner.h"

using namespace cv;

std::vector<KeyPoint> harrisCorner(const Mat& img, int threshold) {
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

	Mat cThreshold = Mat::zeros(cNorm.size(), CV_32F);
	
	int rows = img.rows;
	int cols = img.cols;
	for (int i = 0; i < rows; ++i) {
		for (int j = 0; j < cols; ++j) {
			float intensity = cNorm.at<float>(i, j);
			if (intensity >= threshold) {
				cThreshold.at<float>(i, j)=intensity;
			}
		}
	}

	localMaxima(cThreshold);
	std::vector<KeyPoint> corners;
	for (int i = 0; i < rows; ++i) {
		for (int j = 0; j < cols; ++j) {
			float intensity=cThreshold.at<float>(i, j);
			if (intensity != 0) {
				KeyPoint kPoint(Point(j,i), 5,-1.0F,intensity);
				corners.push_back(kPoint);
			}
		}
	}

	return corners;

}

void localMaxima(cv::Mat& in) {
	for (int i = 1; i < in.rows - 2; ++i) {
		for (int j = 1; j < in.cols - 2; ++j) {
			//loop within 3*3 box
			float max = 0;
			int row=-1, col=-1;
			for (int m = -1; m < 2; ++m) {
				for (int n = -1; n < 2; ++n) {
					float current = in.at<float>(i+m, j+n);
					if (current!=0 && current >= max) {
						max = current;
						//set previous max to 0
						if (row != -1) in.at<float>(row, col) = 0.0F;
						row = i + m;
						col = j + n;
					}
					else {
						in.at<float>(i+m,j+n) = 0.0F;
					}
				}
			}

		}
	}
}