# Project Description:
This project implements 2-D object recognition by identifying a specified set of objects placed on a white surface in a translation, scale, and rotation invariant manner from a camera looking down. This project uses an image directory of 11 different 2D objects of differentiable shape and a uniform dark color. Overall, the project converts input image to binary by thresholding, cleans up the image, gets region by image segmentation, calculates and stores features- HuMoments, width to height ratio, and % area filled, of the image. The program also trains the dataset with features for 11 objects and stores it as .csv file. This trained dataset is then used to classify new images using two different classifiers.

# Requirements
The project is tested in the following environment
- ubuntu 20.04
- VScode 1.74.3
- cmake 3.16.3
- g++ 9.4.0
- opencv 3.4.16

# Links to Demo Video
Training: https://northeastern-my.sharepoint.com/:v:/r/personal/deore_t_northeastern_edu/Documents/PRCV/Project3/Training.mp4?csf=1&web=1&e=aOvcay

Nearest Neighbour classification: https://northeastern-my.sharepoint.com/:v:/r/personal/deore_t_northeastern_edu/Documents/PRCV/Project3/Classification_1.mp4?csf=1&web=1&e=TC0MJS

K-NN classification: https://northeastern-my.sharepoint.com/:v:/r/personal/deore_t_northeastern_edu/Documents/PRCV/Project3/Classification_KNN.mp4?csf=1&web=1&e=gpb4Jh

# Instructions for Running Executables

To run the project executables, follow these steps:

1. Place the following files in a folder named "Project3": CMakeLists.txt, functions.cpp, ImgDB.cpp, csv_util.cpp, csv_util.h, and functions.h.
2. Create a folder named "build" inside "Project3" using the command mkdir build.
3. Navigate to the build folder using the command cd ~/Project3/build.
4. Run cmake using the command cmake ...
5. Run make using the command make.
6. The execution command should contain the following arguments in the given order: path of the training image dataset, path of the classification dataset, "t" for training mode or "c" for classification mode, "x" for nearest-neighbor classification or "y" for KNN classification (e.g., ./Project3 </path of the training image dataset> </path of the classification dataset> c x).
7. In training mode, the user will have the option to select or discard a region. To continue, enter "y"; to quit, enter "q"; to skip the region, enter any other key.
8. The resulting .csv file with feature vectors for corresponding objects will be stored in the current working directory as "FeatureVectors.csv".
9. After training the system, paste the path of the saved .csv file in the code.
10. In classification mode, the user will have the option to continue or skip an image. To continue, enter "y"; to quit, enter "q"; to skip the image, enter any other key.

Objects used: Pen, Mask, Adapter, Watch, Calculator, Comb, Earphones, ZED Box, Cloth, Bottle Cap, Spectacle Case

# Acknowledgements

[1] Professor Bruce Maxwell : author of csv_util files.

[2] Inbuilt Moment functions in open CV. //https://learnopencv.com/shape-matching-using-hu-moments-c-python/

[3] Map data structure. https://www.geeksforgeeks.org/map-associative-containers-the-c-standard-template-library-stl/

[4] Putting a text in image on OpenCv. https://www.geeksforgeeks.org/write-on-an-image-using-opencv-in-cpp/
