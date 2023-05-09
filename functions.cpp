/* 
Author: Tejaswini Dilip Deore 
*/

#include <cstdio>
#include <cstring>
#include <vector>
#include "opencv2/opencv.hpp"
#include <cstdlib>
#include "functions.h"
using namespace cv;

/*
Thresholding the input image and converting to binary image
*/
int thresholding(Mat &src, Mat &thresh){
    Mat blur, dst;
    thresh = Mat( src.size(), CV_8UC1 ); //initialize dst as zero matrix
    //Pre-process the image- Blur image
    GaussianBlur(src,blur,Size(3,3),0,0);
    cvtColor(blur, dst, COLOR_BGR2GRAY);
    for(int i=0; i< src.rows; i++){
        uchar *rptr = dst.ptr<uchar>(i);
        uchar *dptr = thresh.ptr<uchar>(i);
        for(int j=0; j<src.cols; j++){
            //int threshold= (rptr[j+0]+rptr[j+1]+rptr[j+2])/3;
            //for(int c=0;c<3;c++){
                if(rptr[j]<=100){
                    dptr[j]=255;
                }
                else{
                    dptr[j]=0;
                }
           // }                                                                         
        }
    }

    return (0);
}

/*
Clean up binary image using erosion and dilation operations 
Erosion followed by dilation is used to remove noise and fill holes
*/
int cleanup(Mat &src, Mat &dst){
    Mat dilated,erode,thresh;
    dst = Mat( src.size(), CV_8UC1 );
    thresholding(src, thresh);
    erosion_img(thresh,erode,10);
    dilation_img(erode,dilated,20); 
    dilated.copyTo(dst);
    return(0);
}

//Erosion usign 4-connected neighborhood by accessing the pixels in the image and setting the pixel to 0 if any of the 4-connected neighbors are 0
int erosion_img(Mat &src, Mat &dst, int iterations){
    dst = Mat::zeros( src.size(), CV_8UC1);
    //loop over the image and set the pixel to 0 if any of the 4-connected neighbors are 0
    for(int k=0; k<iterations;k++){
        for(int i=1; i<src.rows-1;i++){
            for(int j=1;j<src.cols-1;j++){
                if(src.at<uchar>(i,j)==0){
                    dst.at<uchar>(i-1,j)=0;
                    dst.at<uchar>(i+1,j)=0;
                    dst.at<uchar>(i,j-1)=0;
                    dst.at<uchar>(i,j+1)=0;
                }
                else{
                    dst.at<uchar>(i,j)=src.at<uchar>(i,j);
                }
            }
        }
    }
    return(0);
}

//Dilation usign 8-connected neighborhood by accessing the pixels in the image and setting the pixel to 255 if any of the 8-connected neighbors are 255
int dilation_img(Mat &src, Mat &dst, int iterations){
    dst = Mat::zeros( src.size(), CV_8UC1 );
    // loop over the image and set the pixel to 255 if any of the 8-connected neighbors are 255
    for(int k=0; k<iterations;k++){
        for(int i=1; i<src.rows-1;i++){
            for(int j=1;j<src.cols-1;j++){
                if(src.at<uchar>(i,j)==255){
                    dst.at<uchar>(i-1,j)=255;
                    dst.at<uchar>(i+1,j)=255;
                    dst.at<uchar>(i,j-1)=255;
                    dst.at<uchar>(i,j+1)=255;
                    dst.at<uchar>(i-1,j-1)=255;
                    dst.at<uchar>(i-1,j+1)=255;
                    dst.at<uchar>(i+1,j-1)=255;
                    dst.at<uchar>(i+1,j+1)=255;
                }
                else{
                    dst.at<uchar>(i,j)=src.at<uchar>(i,j);
                }
            }
        }
    }
    return(0);
}

/*
Segmentation of the image using connected components algorithm 
This outputs the labels for each region and the centroids of each region along with labels for the largest 3 regions
*/
int segment_regions(Mat &src, Mat &dst, Mat &labels, Mat &centroids, std::vector<int> &labelArr){ 
    Mat stats,sortedIdx;
    int numLabels = connectedComponentsWithStats(src, labels, stats, centroids);
    std::vector<Vec3b> colors(numLabels);
    colors[0] = Vec3b(0, 0, 0); //assign background color to black
    Mat areas = Mat::zeros(1, numLabels - 1, CV_32S);
    // sort the regions by area
    for(int i = 1; i < numLabels; i++){
        int area= stats.at<int>(i, CC_STAT_AREA);
        areas.at<int>(i-1)=area;
    }
    if (areas.cols > 0) {
        sortIdx(areas, sortedIdx, SORT_EVERY_ROW + SORT_DESCENDING); // sort the areas in descending order
    }
    int N_regions = 3; // take the largest 3 non-background regions
    N_regions = (N_regions < sortedIdx.cols) ? N_regions : sortedIdx.cols; 
    int threshold = 5000; // any region area less than 5000 will be ignored
    for (int i = 0; i < N_regions; i++) {
        int label = sortedIdx.at<int>(i) + 1;
        if (stats.at<int>(label, CC_STAT_AREA) > threshold) {
            colors[label] = Vec3b( (rand()&255), (rand()&255), (rand()&255) );
            labelArr.push_back(label);  // store the labels of the largest 3 regions
        }
    }
    // assign colors to each region
    dst= Mat(src.size(), CV_8UC3);
    for(int r = 0; r < dst.rows; ++r){
        for(int c = 0; c < dst.cols; ++c){
            int label = labels.at<int>(r, c);
            Vec3b &pixel = dst.at<Vec3b>(r, c);
            pixel = colors[label];
        }
    }    
    return(0);
}

/*
Compute features of the segmented regions 
The fecature vector regionFeature contains HuMoments, ratio of height and width of the bounding box, and the % of area filled by the region
This outputs the features of the largest 3 regions
*/
int getFeatures(Mat &src, Mat &labels, Mat &centroids, std::vector<int> &labelArr, std::vector<float> &regionFeature){
    std::vector<float> hu_Moments;
    for(int i=0; i<labelArr.size(); i++){
        int label = labelArr[i];
        Mat regions= (labels==label);
        Moments m = moments(regions, true);
        // get the centroid of the region
        double centroidX = centroids.at<double>(label, 0); 
        double centroidY = centroids.at<double>(label, 1);
        double alpha = 1.0 / 2.0 * atan2(2 * m.mu11, m.mu20 - m.mu02);
        // get the least central axis and bounding box of this region
        RotatedRect boundingBox = BoundingBox(regions, centroidX, centroidY, alpha);
        // draw the centroid and bounding box
        DrawArrow(src, centroidX, centroidY, alpha, Scalar(0, 0, 255));
        DrawBoundingBox(src, boundingBox, Scalar(0, 255, 0));

        regionFeature.clear();
        //Feature1- Hu Moments
        hu_Moments.clear();
        getHuMoments(m, hu_Moments);

        // convert the hu moments to log scale
        for(int i = 0; i < hu_Moments.size(); i++) {
            hu_Moments[i] = -1 * copysign(1.0, hu_Moments[i]) * log10(abs(hu_Moments[i]));
            regionFeature.push_back(hu_Moments[i]); // add the hu moments to the feature vector
        }

        //Feature2 - width to height ratio
        float lengthX=boundingBox.size.width;
        float lengthY=boundingBox.size.height;
        float ratio= (float)lengthX/lengthY;

        //Feature3- % filled area
        float filled= (m.m00)/(lengthX*lengthY);

        //Append ratio and % filled features at the end of HuMoment vector
        regionFeature.push_back(ratio);
        regionFeature.push_back(filled);
    }    
    return(0);
}

/*
Compute bounding box of the region
This outputs the bounding box of the region
*/
RotatedRect BoundingBox(Mat &regions, double centroidX, double centroidY, double alpha){
    int maxX = INT_MIN, minX = INT_MAX, maxY = INT_MIN, minY = INT_MAX;
    for (int i = 0; i < regions.rows; i++) {
        for (int j = 0; j < regions.cols; j++) {
            if (regions.at<uchar>(i, j) == 255) {
                int X = (i - centroidX) * cos(alpha) + (j - centroidY) * sin(alpha); 
                int Y = -(i - centroidX) * sin(alpha) + (j - centroidY) * cos(alpha);
                maxX = max(maxX, X);
                minX = min(minX, X);
                maxY = max(maxY, Y);
                minY = min(minY, Y);
            }
        }
    }
    int lengthX = maxX - minX;
    int lengthY = maxY - minY;
    if(lengthX<lengthY){
        int temp=lengthX;
        lengthX=lengthY;
        lengthY=temp;
    }
    Point centroid = Point(centroidX, centroidY);
    Size size = Size(lengthX, lengthY);
    return RotatedRect(centroid, size, alpha * 180.0 / CV_PI);
}

/*
Draw the bounding box of the region
*/
void DrawBoundingBox(Mat &src, RotatedRect boundingBox, Scalar color){
    Point2f rect[4];
    boundingBox.points(rect);
    for (int i = 0; i < 4; i++) {
        line(src, rect[i], rect[(i + 1) % 4], color, 2);
    }
}

/*
Draw arrow to indicate the orientation of the region
*/
void DrawArrow(Mat &src, double centroidX, double centroidY, double alpha, Scalar color) {
    double length = 90.0;
    double e1 = length * sin(alpha);
    double e2 = sqrt(length * length - e1 * e1);
    double x = centroidX + e2, y = centroidY + e1;
    arrowedLine(src, Point(centroidX, centroidY), Point(x, y), color, 3);
}

// Get Hu Moments
int getHuMoments(Moments m, std::vector<float> &hu_Moments){
    double hu[7];
    HuMoments(m, hu);
    // Array to vector
    for (double v : hu) {
        hu_Moments.push_back(v);
    }
    return(0);
}

//Get distance between two feature vectors using Cosine Similarity 
float CosineSimilarity(std::vector<float> FeatureDB, std::vector<float> featurevect){
    // check if the feature vectors are compatible
    if (FeatureDB.empty() || featurevect.empty() || FeatureDB.size() != featurevect.size()) {
        printf("Feature Vector size incompatible\n");
        return 0.0f; // or any other value that makes sense for your application
    }
    float a=0, b=0, mul=0;
    for(int i=0;i<FeatureDB.size();i++){
        a= a + (FeatureDB[i]*FeatureDB[i]);
        b= b + (featurevect[i]*featurevect[i]);
        mul= mul + ((FeatureDB[i]- featurevect[i])*(FeatureDB[i]- featurevect[i]));
    }
    float result= sqrt(mul)/ (sqrt(a)+sqrt(b)); 
    return(result);
}
/*
* Classifier 1: nearest-neighbor 
* using Cosine Similarity as distance metric
* compare the feature vector of the region with the feature vectors of the images in the trained database
* this returns the label of the image in the database that is most similar to the region
*/
std::string classify_images(std::vector<std::vector<float>> &FeaturesDB, std::vector<float> &regionFeature,std::vector<char *> labelDB){
    float cutoff = 0.15; // cutoff value for similarity
    float dist = DBL_MAX;
    std::string newlabel;
    bool foundMatch= false;
    for(int i=0;i<FeaturesDB.size();i++){
        //std::vector<float> featurevect= FeaturesDB[i];
        std::string label= labelDB[i];
        //std::cout<<labelDB[i]<<std::endl;
        float result= CosineSimilarity(FeaturesDB[i], regionFeature);
        if(result< dist && result<cutoff){
            dist=result;
            newlabel= label;
            foundMatch=true;
        }
    }
    if(!foundMatch){
        newlabel="Unknown";
    }
    return(newlabel);
}

/*
* Classifier 2: K-nearest-neighbor
* using Cosine Similarity as distance metric
* compare the feature vector of the region with the feature vectors of the images in the trained database
* this returns the label of the image in the database that is most similar to the region
*/
std::string KNN_Classifier(std::vector<std::vector<float>> FeaturesDB, std::vector<float> regionFeature,std::vector<char *> labelDB, int k){
    float cutoff = 0.7;
    std::vector<float> distances;
    std::string newlabel;
    bool foundMatch= false;
    //Get the values from distance metrics below cutoff value
    for(int i=0;i<FeaturesDB.size();i++){
        float distance= CosineSimilarity(FeaturesDB[i], regionFeature); // get the distance between the feature vectors using Cosine Similarity
        // check if the distance is less than the cutoff value
        if(distance<cutoff){
           distances.push_back(distance);
        }
    }
    //Sort the distance vector in ascending order
    if (distances.size() > 0) {
        std::vector<int> sortedDist;
        sortIdx(distances, sortedDist, SORT_EVERY_ROW + SORT_ASCENDING); // sort the distance vector in ascending order

        // get the first k labels from the sorted distance vector
        int s = sortedDist.size();
        std::map<std::string, int> LabelCount;
        int range = min(s, k); // get labels less than k 
        for (int i = 0; i < range; i++) {
            std::string name = labelDB[sortedDist[i]];
            // check if the label is already present in the map
            if (LabelCount.find(name) != LabelCount.end()) {
                LabelCount[name]++; //increment the label count if neighbor with same label is present
            } else {
                LabelCount[name] = 1;
            }
        }

        // get the class name that appear the most times in the K nearest neighbors
        int count = 0;
        for (std::map<std::string ,int>::iterator it = LabelCount.begin(); it != LabelCount.end(); it++) {
            if (it->second > count) {
                newlabel = it->first;
                count = it->second;
            }
            foundMatch=true;
        }
    }
    if(!foundMatch){
        newlabel="Unknown";
    }
    return newlabel;
}

//Get the label for the object 
std::string getLabel(char ch) {
    std::map<char, std::string> myMap {
        {'p', "Pen"}, {'m', "Mask"}, {'a', "Adapter"}, {'w', "Watch"}, {'c', "Calculator"}, {'l', "Comb"}, {'e', "Earphones"}, {'z', "ZEDBox"}, {'h', "Cloth"}, {'b', "BottleCap"}, {'s', "SpectacleCase"}
        };
    return myMap[ch];
}

//Put text on the image 
int TextImg(Mat &im, Mat centroids, std::vector<int> labelArr,std::string newlabel){
    for(int i=0;i<labelArr.size();i++){
        int label = labelArr[i];
        putText(im, newlabel, Point(centroids.at<double>(label, 0), centroids.at<double>(label, 1)), FONT_HERSHEY_COMPLEX  , 1, Scalar(0, 0, 255), 3);
    }
    return(0);
}
