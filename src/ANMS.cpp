#include"ANMS.h"

using namespace std;
using namespace cv;

void ANMS(std::vector<KeyPoint>& keyPoint, std::vector<KeyPoint>& corners) {
	const int n = 500;
	const float robustC = 0.9F;

	vector<float> robustResponse;
	vector<KeyPoint> r;


	float globalMax = 0.0F;

	for (KeyPoint point : keyPoint) {
		float response = point.response;
		robustResponse.push_back(response*robustC);
		if (response > globalMax) {
			globalMax = response;
		}
	}

	globalMax *= robustC;

	for (KeyPoint point : keyPoint) {
		float response = point.response;
		if (response > globalMax) {
			r.push_back(KeyPoint(point.pt, 3, -1.0F, numeric_limits<float>::max()));
		}
		else {
			float minDist = numeric_limits<float>::max();
			for (int i = 0; i < keyPoint.size(); ++i) {
				float resP = robustResponse[i];
				if (resP > response) {
					float dist = getDistance(keyPoint[i], point);
					if (dist < minDist) {
						minDist = dist;
					}
				}
			}
			r.push_back(KeyPoint(point.pt, 3, -1.0F, minDist));
		}
	}

	sort(r.begin(), r.end(), compDist);
	if (r.size() > n) {
		corners.insert(corners.begin(), r.begin(), r.begin() + n - 1);
	}
	else {
		corners.insert(corners.begin(), r.begin(), r.end());
	}

}

void ANMS(const cv::Mat& cNorm, std::vector<KeyPoint>& corners, int threshold) {

	vector<KeyPoint> keyPoint;

	for (int i = 0; i < cNorm.rows; ++i) {
		for (int j = 0; j < cNorm.cols; ++j) {
			float response = cNorm.at<float>(i, j);
			if (response <= threshold) continue;
			keyPoint.push_back(KeyPoint(Point(j, i), 3, -1.0F, response));
		}
	}

	ANMS(keyPoint, corners);
}

float getDistance(const KeyPoint& p1, const KeyPoint& p2) {
	float diffX = p1.pt.x - p2.pt.x;
	float diffY = p1.pt.y - p2.pt.y;
	float sum = diffX * diffX + diffY * diffY;
	return sqrtf(sum);
}

bool compDist(const KeyPoint& p1, const KeyPoint& p2) {
	return p1.response > p2.response;
}