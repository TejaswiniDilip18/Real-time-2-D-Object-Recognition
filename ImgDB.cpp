/* 
Author: Tejaswini Dilip Deore 
*/

#include <opencv2/opencv.hpp>  // OpenCV main include file
#include <iostream>
#include <cstdio>
#include <cstring>
#include <dirent.h>
#include "functions.h"
#include "csv_util.h"

int main( int argc, char *argv[] )
{
    cv::Mat image, thresh, erode, cleaned_img, segmented_img, labels, features, originalimage;
    bool training=false, classify= false;
    std::vector<float> regionFeature;
    std::vector<char *> labelDB;
    std::vector<std::vector<float>> FeaturesDB;
    char dirname[256], choice[256],buffer[256],file[256], filename[256], labelname[256]={}; 
    FILE *fp;
    DIR *dirp;
    struct dirent *dp;

    //Print a menu to the user
    //printf("Menu:\nt: Threshold image\nc: Clean the image\ns: Segment the image into regions\nf: Compute features for each major region\nn: Collect training data\nm: Classify new images\nq: Exit\n");

    // check for sufficient arguments
    if( argc < 5) {
    printf("usage: %s <directory path>\n", argv[0]);
    exit(-1);
    }
    // get the operation to be performed
    if( !strcmp(argv[3],"t")){
        printf("Training mode\n");
        strcpy(dirname, argv[1]);
        training=true;
    }
    else if(!strcmp(argv[3],"c")){
        printf("Classify mode\n");
        strcpy(dirname, argv[2]);
        classify=true;
    }
    
    strcpy(choice, argv[4]); 
    printf("choice: %s\n", choice);

    printf("Processing directory %s\n", dirname );

    // open the directory
    dirp = opendir( dirname );
    if( dirp == NULL) {
        printf("Cannot open directory %s\n", dirname);
        exit(-1);
    }
    
    strcpy(filename,"FeatureVectors");
    strcpy(file, dirname);
    strcpy(file, filename);
    strcat(file, ".csv");

    // loop over all the files in the image file listing
    while( (dp = readdir(dirp)) != NULL) {
        // check if the file is an image
        if( strstr(dp->d_name, ".jpg") ||
        strstr(dp->d_name, ".png") ||
        strstr(dp->d_name, ".ppm") ||
        strstr(dp->d_name, ".tif") ||
        strstr(dp->d_name, ".jpeg")) {

        // build the overall filename
        strcpy(buffer, dirname);
        strcat(buffer, "/");
        strcat(buffer, dp->d_name);
        // printf("full path name: %s\n", buffer);

        image= cv::imread(buffer);
        image.copyTo(originalimage);

        cv::namedWindow("Original Image", cv::WINDOW_KEEPRATIO);
        cv::imshow("Original Image", image);

        //thresholding the image to get binary image
        thresholding(image, thresh);
        cv::namedWindow("After Thresholding", cv::WINDOW_KEEPRATIO);
        cv::imshow("After Thresholding", thresh);

        //Use morphological operations to clean the image
        cleanup(image, cleaned_img);
        cv::namedWindow("Cleaned Image", cv::WINDOW_KEEPRATIO);
        cv::imshow("Cleaned Image", cleaned_img);
        
        //Segment the image into regions
        std::vector<int> labelArr;
        cv::Mat centroids;
        segment_regions(cleaned_img, segmented_img, labels, centroids, labelArr);
        //printf("Valid regions: %ld\n", labelArr.size());
        // cv::namedWindow("Connected Components", cv::WINDOW_KEEPRATIO);
        // cv::imshow("Connected Components", segmented_img);

        //Find features for each major region
        std::vector<float> hu_Moments;
        for(int i=0; i<labelArr.size(); i++){
            int label = labelArr[i];
            cv::Mat regions;
            regions = (labels==label); //get the region of interest
            Moments m = moments(regions, true); //get the moments of the region
            //get the centroid of the region
            double centroidX = centroids.at<double>(label, 0); 
            double centroidY = centroids.at<double>(label, 1);
            double alpha = 0.5 * atan2(2 * m.mu11, m.mu20 - m.mu02); 
            // get the least central axis and bounding box of this region
            RotatedRect boundingBox = BoundingBox(regions, centroidX, centroidY, alpha);
            //DrawArrow(image, centroidX, centroidY, alpha, Scalar(0, 0, 255));
            DrawBoundingBox(image, boundingBox, Scalar(0, 255, 0));

            //Feature1- Hu Moments
            regionFeature.clear();
            hu_Moments.clear();
            getHuMoments(m, hu_Moments);

            // convert the hu moments to log scale
            for(int i = 0; i < 4; i++) {
                hu_Moments[i] = -1 * copysign(1.0, hu_Moments[i]) * log10(abs(hu_Moments[i]));
                regionFeature.push_back(hu_Moments[i]); // add the hu moments to the feature vector
            }

            //Feature2 - width to height ratio
            float lengthX=boundingBox.size.width;
            float lengthY=boundingBox.size.height;
            float ratio= (float)lengthX/lengthY;

            //Feature3- % area filled
            float filled= (m.m00)/(lengthX*lengthY);

            //Append ratio and % filled features at the end of HuMoment vector
            regionFeature.push_back(ratio);
            regionFeature.push_back(filled);
            
            // Enter training mode
            if(training){
                cv::namedWindow("Current Region", cv::WINDOW_KEEPRATIO); //create a window to display the current region
                cv::imshow("Current Region", regions); 
                // Print the label menu
                printf("Press y to continue\nPress q to quit\nPress any other key to skip the region\n");
                char key = cv::waitKey(0);
                if(key=='q'){
                    return(-1);
                }
                else if(key=='y'){
                cv::namedWindow("Image", cv::WINDOW_KEEPRATIO); //create a window to display the current region
                // Draw arrow to show the orientation of the region
                DrawArrow(image, centroidX, centroidY, alpha, Scalar(0, 0, 255));
                cv::imshow("Image", image);
                printf("Input label for the object:\np: Pen\nm: Mask\na: Adapter\nw: Watch\nc: Calculator\nl: Comb\ne: Earphones\nz: ZEDBox\nh: Cloth\nb: BottleCap\ns: SpectacleCase\nq: Quit\n");
                char ch = waitKey(0);
                std::string name = getLabel(ch); // get the label 
                strcpy(labelname, name.c_str());
                // Write the computed features to a CSV file
                append_image_data_csv(file,labelname, regionFeature, 0);
                    if(ch=='q'){
                        return(-1);
                    }
                }
               // cv::destroyWindow("Image");
                cv::destroyWindow("Current Region");
                
            }
            // Enter classification mode
            else if(classify){
                std::string newlabel;
                strcpy(file, "/home/tejaswini/PRCV/Project3/build/FeatureVectors.csv"); // Path to get feature vectors from training data
                read_image_data_csv(file,labelDB, FeaturesDB, 0); // read the feature vectors from the CSV file
                // Draw arrow to show the orientation of the region
                DrawArrow(image, centroidX, centroidY, alpha, Scalar(0, 0, 255));

                cv::namedWindow("Original Image", cv::WINDOW_KEEPRATIO); //create a window to display the current region
                
                printf("Press c to continue classification\nPress q to quit\nPress any other key to skip the image\n");
                char key = cv::waitKey(0);
                if(key=='q'){
                    return(-1);
                }
                else if (key=='c'){ 
                    if(strcmp(choice, "x")==0){
                        newlabel = classify_images(FeaturesDB,regionFeature,labelDB); // Classify using the CosineSimilarity
                        std::cout<<"newlabel: "<<newlabel<<std::endl;
                    }
                    else if(strcmp(choice, "y")==0){
                        newlabel = KNN_Classifier(FeaturesDB,regionFeature,labelDB,3); // Classify using KNN matching
                    }
                    //printf("After classification\n");
                    putText(image, newlabel, Point(centroids.at<double>(label, 0), centroids.at<double>(label, 1)), cv::FONT_HERSHEY_COMPLEX  , 1.5, Scalar(0, 0, 255), 3);
                    cv::namedWindow("Image", cv::WINDOW_KEEPRATIO);
                    cv::imshow("Original Image", originalimage);
                    cv::imshow("Image", image);
                }
            }
            else{
            cv::namedWindow("Original Image", cv::WINDOW_AUTOSIZE);
            cv::imshow("Original Image", image);    
        }                            
    }
    }
    }
    cv::destroyWindow("Image");
    return 0;
}
