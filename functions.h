/* 
Author: Tejaswini Dilip Deore 
*/

#include <opencv2/opencv.hpp>
using namespace cv;

int thresholding(Mat &src, Mat &dst);
int cleanup(Mat &src, Mat &dst);
int erosion_img(Mat &src, Mat &dst, int iterations);
int dilation_img(Mat &src, Mat &dst, int iterations);
int segment_regions(Mat &src, Mat &dst, Mat &labels, Mat &centroids, std::vector<int> &labelArr);
int getFeatures(Mat &src, Mat &regions, Mat &centroids, std::vector<int> &labelArr, std::vector<float> &regionFeature);
RotatedRect BoundingBox(Mat &regions, double centroidX, double centroidY, double alpha);
void DrawBoundingBox(Mat &src, RotatedRect boundingBox, Scalar color);
void DrawArrow(Mat &src, double centroidX, double centroidY, double alpha, Scalar color);
int getHuMoments(Moments m, std::vector<float> &hu_Moments);
float CosineSimilarity(std::vector<float> FeatureDB, std::vector<float> featurevect);
std::string classify_images(std::vector<std::vector<float>> &FeaturesDB, std::vector<float> &regionFeature,std::vector<char *> labelDB);
std::string KNN_Classifier(std::vector<std::vector<float>> FeaturesDB, std::vector<float> regionFeature,std::vector<char *> labelDB, int k);
std::string getLabel(char ch);
int TextImg(Mat &im, Mat centroids, std::vector<int> labelArr,std::string newlabel);
