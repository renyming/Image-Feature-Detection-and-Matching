#include <opencv2/opencv.hpp>
#include <vector>
#include "HarrisCorner.h"

using namespace std;
using namespace cv;

int main(int argc, char* argv[]) {

	String fileName = argv[1];
	int threshold = stoi(argv[2]);

	Mat origImage = imread(fileName);
	vector<KeyPoint> featurePoints = harrisCorner(origImage,threshold);

	Mat featurePointsImg;
	drawKeypoints(origImage, featurePoints, featurePointsImg);
	imshow("", featurePointsImg);
	waitKey();
	imwrite("result.jpg", featurePointsImg);

}