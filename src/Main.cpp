#include <opencv2/opencv.hpp>
#include <vector>
#include "HarrisCorner.h"
#include "GenerateDescriptor.h"
#include "SSD.h"

using namespace std;
using namespace cv;

int main(int argc, char* argv[]) {

	String fileName1 = argv[1];
	String fileName2 = argv[2];
	int thresholdFeature = stoi(argv[3]);
	int thresholdMatch = stoi(argv[4]);
	int useANMS = stoi(argv[5]);
	bool isANMS = (useANMS > 0) ? true : false;

	Mat origImage1 = imread(fileName1);
	vector<KeyPoint> featurePoints1;
	harrisCorner(origImage1, featurePoints1, thresholdFeature,isANMS);

	Mat origImage2 = imread(fileName2);
	vector<KeyPoint> featurePoints2;
	harrisCorner(origImage2, featurePoints2, thresholdFeature,isANMS);

	Mat featurePointsImg;
	drawKeypoints(origImage1, featurePoints1, featurePointsImg);
	imshow("Key Points", featurePointsImg);
	waitKey();

	drawKeypoints(origImage2, featurePoints2, featurePointsImg);
	imshow("Key Points", featurePointsImg);
	waitKey();

	vector<Descriptor> keyPointDescriptor1;
	generateDescriptor(origImage1, featurePoints1, keyPointDescriptor1);

	vector<Descriptor> keyPointDescriptor2;
	generateDescriptor(origImage2, featurePoints2, keyPointDescriptor2);

	vector<KeyPoint> keyPoint1;
	vector<KeyPoint> keyPoint2;
	vector<DMatch> dMatch;
	findMatch(keyPointDescriptor1, keyPointDescriptor2, keyPoint1, keyPoint2, dMatch, thresholdMatch);

	Mat out;
	drawMatches(origImage1, keyPoint1, origImage2, keyPoint2, dMatch,out);
	imshow("Match", out);
	waitKey();
}