#include "GenerateDescriptor.h"

using namespace std;
using namespace cv;

void generateDescriptor(const Mat& in, const vector<KeyPoint> keyPoint, vector<Descriptor>& keyPointDescriptor) {
	Mat grayImg;
	cvtColor(in, grayImg, COLOR_BGR2GRAY);

	vector<KeyPoint> keyPointRotation;
	kPRotation(grayImg, keyPoint, keyPointRotation);

	extractDescriptor(grayImg, keyPointRotation, keyPointDescriptor);

}

void extractDescriptor(const Mat& in, const vector<KeyPoint>& keyPointRotation, vector<Descriptor>& keyPointDescriptor) {
	int wSize = 18;
	int halfWSize = (wSize - 2) / 2;
	int nBin = 36;
	int intervalBin = 360 / nBin;
	Mat gKernel;
	getGaussianWeight(gKernel, wSize-2);
	
	for (KeyPoint point : keyPointRotation) {
		//discard key points to avoid corner conditions
		if (point.pt.x <= halfWSize || point.pt.x >= in.cols - halfWSize - 1 || point.pt.y <= halfWSize || point.pt.y >= in.rows - halfWSize - 1) continue;

		Rect patch = Rect(point.pt.x - halfWSize - 1, point.pt.y - halfWSize - 1, wSize, wSize);
		Mat window = Mat(in, patch);
		window.convertTo(window, CV_32F);
		Mat M = Mat(wSize-2, wSize-2, CV_32F);
		Mat angle = Mat(wSize-2, wSize-2, CV_32F);

		for (int i = 1; i < wSize-1; ++i) {
			for (int j = 1; j < wSize-1; ++j) {
				float diffI = window.at<float>(i + 1, j) - window.at<float>(i - 1, j);
				float diffJ = window.at<float>(i, j + 1) - window.at<float>(i, j - 1);
				float m = sqrt(pow(diffI, 2.0) + pow(diffJ, 2.0));
				M.at<float>(i - 1, j - 1) = m *gKernel.at<float>(i - 1, j - 1);
				float theta = atan2(diffI, diffJ) * 180 / M_PI;
				if (theta < 0) theta += 360;

				//normalize rotation
				theta -= point.angle;
				if (theta < 0) theta += 360;
				angle.at<float>(i - 1, j - 1) = theta;
			}
		}

		vector<float> bin;

		for (int i = 0; i < wSize-2; i+=4) {
			for (int j = 0; j < wSize-2; j+=4) {
				vector<float> gridBin(8,0);
				for (int m = 0; m < 4; ++m) {
					for (int n = 0; n < 4; ++n) {
						int pI = i + m;
						int pJ = j + n;
						int binIdx = static_cast<int>(angle.at<float>(pI , pJ) / 45);
						gridBin[binIdx] += M.at<float>(pI,pJ);
					}
				}
				bin.insert(bin.end(), gridBin.begin(), gridBin.end());
			}
		}

		vector<float> binNorm;
		normalize(bin, binNorm, 1.0, 0.0, NORM_L2);

		for (int i = 0; i < binNorm.size(); ++i) {
			if (binNorm[i] > 0.2) binNorm[i] = 0.2;
		}

		normalize(binNorm, bin, 1.0, 0.0, NORM_L2);

		Mat matBin = Mat(1, 128, CV_32F);
		for (int i = 0; i < 128; ++i) {
			matBin.at<float>(i) = bin[i];
		}

		keyPointDescriptor.push_back(Descriptor(point.pt.x, point.pt.y, matBin));
	}

}

void kPRotation(const Mat& in, const vector<KeyPoint>& vKeyPoint, vector<KeyPoint>& vKeyPointRotation) {

	//window size for rotation=actual ws+3
	int wSize = 23;
	int halfWSize = (wSize-2) / 2;
	int nBin = 36;
	int intervalBin = 360 / nBin;

	Size kSize = getKSize(wSize);
	float sigma = getSigma(wSize);

	for (KeyPoint point : vKeyPoint) {

		//discard key points to avoid corner conditions
		if (point.pt.x <= halfWSize || point.pt.x >= in.cols - halfWSize - 1 || point.pt.y <= halfWSize || point.pt.y >= in.rows - halfWSize - 1) continue;

		Rect patch = Rect(point.pt.x - halfWSize - 1, point.pt.y - halfWSize - 1, wSize, wSize);
		Mat window = Mat(in, patch);
		Mat gWindow;
		gWindow = window;
		GaussianBlur(window, gWindow, kSize, sigma);
		
		vector<float> bin(36, 0);

		for (int i = 1; i < gWindow.rows-1; ++i) {
			for (int j = 1; j < gWindow.cols-1; ++j) {
				int diffI = gWindow.at<uchar>(i + 1, j) - gWindow.at<uchar>(i - 1, j);
				int diffJ = gWindow.at<uchar>(i, j + 1) - gWindow.at<uchar>(i, j - 1);
				float m = sqrt(pow(diffI, 2.0) + pow(diffJ, 2.0));
				float theta = atan2(diffI, diffJ) * 180 / M_PI;
				if (theta < 0) theta += 360;
				int binIdx = static_cast<int>(theta / intervalBin);
				bin[binIdx] += m;
			}
		}

		normalize(bin, bin, 0, 100, NORM_MINMAX);

		//float max = *max_element(begin(bin), end(bin));
		for (int i = 0; i <nBin; ++i) {
			//bin[i] /= max;
			if (bin[i] >80) {
				vKeyPointRotation.push_back(KeyPoint(point.pt, 15, i*intervalBin,point.response));
			}
		}
	}
}

Size getKSize(int wSize) {
	float sigma = getSigma(wSize);
	int size=((sigma - 0.8) / 0.3 + 1) / 0.5 + 1;
	if (size % 2 == 0) size += 1;
	return Size(size, size);
}

float getSigma(int wSize) {
	return wSize / 1.5;
}

void normVector(vector<float>& v) {
	float sum = 0.0F;
	for (int i = 0; i < v.size(); ++i) {
		sum += v[i] * v[i];
	}
	sum = sqrtf(sum);
	for (int i = 0; i < v.size(); ++i) {
		v[i] /= sum;
	}
}

void getGaussianWeight(Mat& gaussianWeight, int wSize) {
	float sigma = getSigma(wSize);
	Mat k = getGaussianKernel(wSize, sigma, CV_32F);
	gaussianWeight = k * k.t();
}