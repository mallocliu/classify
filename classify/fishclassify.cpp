#include "stdafx.h"
using namespace cv;

const int generateTestId(const int a, const int b) noexcept{
	srand((unsigned int)(time(NULL)));
	return rand() % (b - a + 1) + a;
}

const std::string imageName(const fishclassify& f, 
							const int fishType, 
							const int imageid) noexcept{
	std::ostringstream id;
	id << imageid << ".jpg";
	return f.baseImgPath + f.Fish[fishType] + id.str();
}

void fishclassify::loadImg() {
	std::cout << "loading image" << std::endl;
	for (int fish = 0; fish < fishNum; ++fish) {
		//generate random test id for this fish
		std::vector<int> testId = {generateTestId(1, imageNum)};
		std::vector<bool> testIdSelected(imageNum, false);
		testIdSelected[testId[0] - 1] = true;
		for (int i = 1; i < testNum; ++i) {
			int x = generateTestId(1, imageNum);
			while (testIdSelected[x - 1]) x = generateTestId(1, imageNum);
			testId.push_back(x);
			testIdSelected[x - 1] = true;
		}

		//load test image
		std::cout << "loading test image of fish " << fish << std::endl;
		for (auto c : testId) {
			std::cout << c << " ";
			std::string name = imageName(*this, fish, c);
			Mat image = imread(name, CV_LOAD_IMAGE_GRAYSCALE);
			testImg.push_back(image);
			testLabel.push_back(fish);
		}
		std::cout << std::endl;
		// load train image
		std::cout << "loading train image of fish " << fish << std::endl;
		for (int i = 1; i != imageNum; ++i) {
			if (!testIdSelected[i - 1]) {
				std::cout << i << " ";
				std::string name = imageName(*this, fish, i);
				Mat image = imread(name, CV_LOAD_IMAGE_GRAYSCALE);
				trainImg.push_back(image);
				trainLabel.push_back(fish);
			}
		}
		std::cout << std::endl;
	}
}

void fishclassify::getfeatureUnclustered() {
	Ptr<Feature2D> sift = xfeatures2d::SIFT::create();
	std::vector<KeyPoint> keypoints;
	Mat descriptor;
	for (auto c : trainImg) {
		sift->detect(c, keypoints);
		sift->compute(c, keypoints, descriptor);
		featureUnclustered.push_back(descriptor);
	}
}

void fishclassify::buildDict() {
	TermCriteria tc(CV_TERMCRIT_ITER, 100, 0.001);
	int retries = 1;
	//necessary flags
	int flags = KMEANS_PP_CENTERS;
	//Create the BoW (or BoF) trainer
	BOWKMeansTrainer bowTrainer(dictionarySize, tc, retries, flags);
	//cluster the feature vectors
	dictionary = bowTrainer.cluster(featureUnclustered);
}

const Mat fishclassify::calcBofDescriptor(const Mat& dictionary, const Mat& image) {
	//create a nearest neighbor matcher
	Ptr<DescriptorMatcher> matcher = FlannBasedMatcher::create();
	//create Sift feature point extracter
	Ptr<FeatureDetector> detector = xfeatures2d::SiftFeatureDetector::create();
	//create Sift descriptor extractor
	Ptr<DescriptorExtractor> extractor = xfeatures2d::SiftDescriptorExtractor::create();
	//create BoF (or BoW) descriptor extractor
	BOWImgDescriptorExtractor bowDE(extractor, matcher);
	bowDE.setVocabulary(dictionary);
	std::vector<KeyPoint> keypoints;
	detector->detect(image, keypoints);
	Mat bowDescriptor;
	bowDE.compute(image, keypoints, bowDescriptor);
	return bowDescriptor;
}

void fishclassify::classification() {
	loadImg();
	getfeatureUnclustered();
	buildDict();
	using ml::SVM;
	using ml::TrainData;
	Ptr<SVM> svm = SVM::create();
	svm->setType(SVM::C_SVC);
	svm->setKernel(SVM::RBF);
	svm->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER, 500, 1e-6));

	Mat trainData;
	for (auto c : trainImg) {
		Mat tmp = calcBofDescriptor(dictionary, c);
		trainData.push_back(tmp);
	}
	trainLabel.convertTo(trainLabel, CV_32SC1);

	Ptr<TrainData> tData = TrainData::create(trainData, ml::SampleTypes::ROW_SAMPLE, trainLabel);
	svm->train(tData);

	Mat testData;
	Mat result;
	for (auto c : testImg) {
		Mat tmp = calcBofDescriptor(dictionary, c);
		testData.push_back(tmp);
	}
	svm->predict(testData, result);

	double error = 0;
	for (int x = 0; x < result.rows; ++x)
		for (int y = 0; y < result.cols; ++y) {
			int id = x*result.cols + y;
			std::cout << "testlabel of " << id + 1 << ": "
				<< result.at<float>(x, y) << std::endl;
			if (static_cast<int>(result.at<float>(x, y)) != (testLabel.at<uchar>(x,y))) error += 1;
		}
	std::cout << "the accuracy is: "
		<< (1 - error / static_cast<double>(fishNum * testNum)) * 100
		<< "%\n" << std::endl;
}