#include"HarrisCorner.h"

using namespace cv;

std::vector<KeyPoint> harrisCorner(const Mat& img, int threshold) {
	Mat greyImg;
	cvtColor(img, greyImg, COLOR_BGR2GRAY);
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

	Mat trace;
	pow(GIx2 + GIy2, 2.0, trace);

	Mat c, cNorm;
	c = det/(GIx2+GIy2);
	//c = det - 0.04*trace;

	normalize(c, cNorm, 0, 255, NORM_MINMAX, CV_32F);
	
	int rows = img.rows;
	int cols = img.cols;
	std::vector<KeyPoint> corners;
	for (int i = 0; i < rows; ++i) {
		for (int j = 0; j < cols; ++j) {
			int intensity = cNorm.at<float>(i, j);
			if (intensity >= threshold)
				corners.push_back(KeyPoint(j, i, 5.0));
		}
	}

	return corners;

}