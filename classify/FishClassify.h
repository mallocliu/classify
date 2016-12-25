#pragma once
#ifndef FISH_CLASSIFY_H
#define FISH_CLASSIFY_H
#include <opencv2/xfeatures2d/nonfree.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/ml/ml.hpp>
#include <vector>
#include <string>
#include <ctime>
#include <cstdlib>
#include <sstream>
class fishclassify {
public:
	friend inline const std::string imageName(const fishclassify&, const int, const int) noexcept;
	const int fishNum = 5;
	const int trainNum = 10;
	const int imageNum = 12;
	const int testNum = imageNum - trainNum;
	const int K = imageNum / testNum; //stands for K,k cross-validation.
	const int dictionarySize = 200; //we build a BOF,and this is the size of dict.
	const std::vector<std::string> Fish = { "ˆ”„", "ˆÙ”„", "Ω«π”„", "Ω”„", "«Ôµ∂”„" };
	const std::string baseImgPath = "D:/Image/";
	fishclassify() = default;
	~fishclassify() {};
	void classification();
	
private:
	void loadImg();
	void getfeatureUnclustered();
	void buildDict();
	const cv::Mat calcBofDescriptor(const cv::Mat&, const cv::Mat&);
	std::vector<cv::Mat> trainImg, testImg;
	cv::Mat trainLabel, testLabel;
	cv::Mat featureUnclustered;
	cv::Mat dictionary;

};

inline const int generateTestId(const int, const int) noexcept;
inline const std::string imageName(const fishclassify&, const int, const int) noexcept;

#endif // !FISH_CLASSIFY_H