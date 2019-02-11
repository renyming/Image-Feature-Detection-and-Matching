#include"SSD.h"

using namespace std;
using namespace cv;

void findMatch(vector<Descriptor>& keyPointDescriptor1, vector<Descriptor>& keyPointDescriptor2, vector<KeyPoint>& keyPoint1, vector<KeyPoint>& keyPoint2, vector<DMatch>& dMatch) {
	int idx = 0;
	for (Descriptor desp1 : keyPointDescriptor1) {
		Point bestPoint1;
		Point bestPoint2;
		float bestDist = numeric_limits<float>::max();
		Point secondPoint1;
		Point secondPoint2;
		float secondDist = bestDist;
		for (Descriptor desp2 : keyPointDescriptor2) {
			float dist = getDistance(desp1.v, desp2.v);
			
			if (dist < bestDist) {
				secondDist = bestDist;
				secondPoint1 = bestPoint1;
				secondPoint2 = bestPoint2;
				bestDist = dist;
				bestPoint1 = Point(desp1.x, desp1.y);
				bestPoint2 = Point(desp2.x, desp2.y);
			}
			else if (dist < secondDist) {
				secondDist = dist;
				secondPoint1 = Point(desp1.x, desp1.y);
				secondPoint2 = Point(desp2.x, desp2.y);
			}
		}
		if (bestDist > 2) continue;
		float ratio = bestDist / secondDist;
		if (ratio <= 0.8) {
			keyPoint1.push_back(KeyPoint(bestPoint1, 5));
			keyPoint2.push_back(KeyPoint(bestPoint2, 5));
			dMatch.push_back(DMatch(idx, idx, 3));
			++idx;
		}
	}
}

float getDistance(Mat& v1, Mat& v2) {
	Mat sub = v1 - v2;
	pow(sub, 2.0, sub);
	float sum=cv::sum(sub)[0];
	return sqrt(sum);
}
