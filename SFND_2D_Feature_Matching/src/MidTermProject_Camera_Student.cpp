/* INCLUDES FOR THIS PROJECT */
#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <vector>
#include <cmath>
#include <limits>
#include <opencv2/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp>

#include "dataStructures.h"
#include "matching2D.hpp"

using namespace std;

/* MAIN PROGRAM */
int main(int argc, const char *argv[])
{

    /* INIT VARIABLES AND DATA STRUCTURES */

    // data location
    string dataPath = "../";

    // camera
    string imgBasePath = dataPath + "images/";
    string imgPrefix = "KITTI/2011_09_26/image_00/data/000000"; // left camera, color
    string imgFileType = ".png";
    int imgStartIndex = 0; // first file index to load (assumes Lidar and camera names have identical naming convention)
    int imgEndIndex = 9;   // last file index to load
    int imgFillWidth = 4;  // no. of digits which make up the file index (e.g. img-0001.png)

    // misc
    int dataBufferSize = 2;       // no. of images which are held in memory (ring buffer) at the same time
    vector<DataFrame> dataBuffer; // list of data frames which are held in memory at the same time
    bool bVis = false;            // visualize results

    /* MAIN LOOP OVER ALL IMAGES */

    string detectorType;
    cout<<"input detector type:SHITOMASI, HARRIS, FAST, BRISK, ORB, AKAZE or SIFT"<<endl; //HARRIS, FAST, BRISK, ORB, AKAZE, SIFT
    cin>>detectorType;

    string descriptorType; // BRIEF, ORB, FREAK, AKAZE, SIFT,BRISK (the default value)
    cout<<"input descritptor type: BRISK, BRIEF, ORB, FREAK, AKAZE or SIFT"<<endl;
    cin>>descriptorType;

    string matcherType;        // MAT_BF, MAT_FLANN
    cout<<"input mactherType: MAT_BF or MAT_FLANN"<<endl;
    cin>>matcherType;

    string match_descriptorType; // DES_BINARY, DES_HOG
    cout<<"input descriptorType=DES_BINARY or DES_HOG"<<endl;
    cin>>match_descriptorType;

    string selectorType = "SEL_KNN";       // SEL_NN, SEL_KNN
    cout<<"selectorType: SEL_KNN or SEL_NN"<<endl;
    cin>>selectorType;

    vector<int> keypointNumber_frames;
    vector<float> keypointExtTime_frames;
    float keypointExtTime;
    float DescripExtTime;
    vector<float> DescripExtTime_frames;
    float DescripExtTime_Sum=0.0;
    int keypointMatched_Num=0;
    //int imgIndex;
    for(int imgIndex = 0;imgIndex<= imgEndIndex - imgStartIndex; imgIndex++)
    {
        /* LOAD IMAGE INTO BUFFER */

        // assemble filenames for current index
        ostringstream imgNumber;
        imgNumber << setfill('0') << setw(imgFillWidth) << imgStartIndex + imgIndex;
        string imgFullFilename = imgBasePath + imgPrefix + imgNumber.str() + imgFileType;

        // load image from file and convert to grayscale
        cout<<"                       "<<endl;
        cout<<"image index  "<<imgIndex<<endl;
        cv::Mat img, imgGray;
        img = cv::imread(imgFullFilename);
        cv::cvtColor(img, imgGray, cv::COLOR_BGR2GRAY);

        //// STUDENT ASSIGNMENT
        //// TASK MP.1 -> replace the following code with ring buffer of size dataBufferSize

        // push image into data frame buffer
        DataFrame frame;
        frame.cameraImg = imgGray;
        if(dataBuffer.size()>=dataBufferSize)
            {dataBuffer.erase(dataBuffer.begin());}
        dataBuffer.push_back(frame);
        cout<<"Data Fuffer size= "<<dataBuffer.size()<<endl;

        //// EOF STUDENT ASSIGNMENT
        cout << "#1 : LOAD IMAGE INTO BUFFER done" << endl;

        /* DETECT IMAGE KEYPOINTS */

        // extract 2D keypoints from current image
        vector<cv::KeyPoint> keypoints; // create empty feature list for current image
        //string detectorType = "SHITOMASI";

        //// STUDENT ASSIGNMENT
        //// TASK MP.2 -> add the following keypoint detectors in file matching2D.cpp and enable string-based selection based on detectorType
        //// -> HARRIS, FAST, BRISK, ORB, AKAZE, SIFT

        if (detectorType.compare("SHITOMASI") == 0)
        {
            keypointExtTime=detKeypointsShiTomasi(keypoints, imgGray, false);
            keypointExtTime_frames.push_back(keypointExtTime);
        }
        else
        {
            //...
            if(detectorType.compare("HARRIS")==0){
            keypointExtTime=detKeypointsHarris(keypoints, imgGray, false);
            }
            else{
            keypointExtTime=detKeypointsModern(keypoints, imgGray, detectorType, false);
            }
            keypointExtTime_frames.push_back(keypointExtTime);
           // descKeypoints(keypoints, imgGray, descriptors, descriptorType)
        }
        keypointNumber_frames.push_back(keypoints.size());
        //// EOF STUDENT ASSIGNMENT

        //// STUDENT ASSIGNMENT
        //// TASK MP.3 -> only keep keypoints on the preceding vehicle

        // only keep keypoints on the preceding vehicle
        bool bFocusOnVehicle = true;
        vector<cv::KeyPoint> keypointsTemp;
        cv::Rect vehicleRect(535, 180, 180, 150);
        if (bFocusOnVehicle)
        { for(auto keypoint:keypoints){
            float x=keypoint.pt.x;
            float y=keypoint.pt.y;
            if(x<vehicleRect.x+vehicleRect.width && x>vehicleRect.x && y<vehicleRect.y+vehicleRect.height && y>vehicleRect.y)
            { keypointsTemp.push_back(keypoint);
            }
            
        }
        keypoints=keypointsTemp;
            // ...
        }

        //// EOF STUDENT ASSIGNMENT

        // optional : limit number of keypoints (helpful for debugging and learning)
        bool bLimitKpts = false;
        if (bLimitKpts)
        {
            int maxKeypoints = 50;

            if (detectorType.compare("SHITOMASI") == 0)
            { // there is no response info, so keep the first 50 as they are sorted in descending quality order
                keypoints.erase(keypoints.begin() + maxKeypoints, keypoints.end());
            }
            cv::KeyPointsFilter::retainBest(keypoints, maxKeypoints);
            cout << " NOTE: Keypoints have been limited!" << endl;
        }

        // push keypoints and descriptor for current frame to end of data buffer
        (dataBuffer.end() - 1)->keypoints = keypoints;
        cout << "#2 : DETECT KEYPOINTS done" << endl;

        /* EXTRACT KEYPOINT DESCRIPTORS */

        //// STUDENT ASSIGNMENT
        //// TASK MP.4 -> add the following descriptors in file matching2D.cpp and enable string-based selection based on descriptorType
        //// -> BRIEF, ORB, FREAK, AKAZE, SIFT

        cv::Mat descriptors;
        //string descriptorType = "SIFT"; // BRIEF, ORB, FREAK, AKAZE, SIFT,BRISK (the default value)
        DescripExtTime=descKeypoints((dataBuffer.end() - 1)->keypoints, (dataBuffer.end() - 1)->cameraImg, descriptors, descriptorType);
        DescripExtTime_frames.push_back(DescripExtTime);
        DescripExtTime_Sum+=DescripExtTime;
        cout<<"Descriptor Executation Time="<<DescripExtTime<<endl;
        //// EOF STUDENT ASSIGNMENT

        // push descriptors for current frame to end of data buffer
        (dataBuffer.end() - 1)->descriptors = descriptors;

        cout << "#3 : EXTRACT DESCRIPTORS done" << endl;

        if (dataBuffer.size() > 1) // wait until at least two images have been processed
        {

            /* MATCH KEYPOINT DESCRIPTORS */

            vector<cv::DMatch> matches;
            //string matcherType = "MAT_FLANN";        // MAT_BF, MAT_FLANN
            //string descriptorType = "DES_BINARY"; // DES_BINARY, DES_HOG
            //string selectorType = "SEL_KNN";       // SEL_NN, SEL_KNN

            //// STUDENT ASSIGNMENT
            //// TASK MP.5 -> add FLANN matching in file matching2D.cpp
            //// TASK MP.6 -> add KNN match selection and perform descriptor distance ratio filtering with t=0.8 in file matching2D.cpp

            /*matchDescriptors((dataBuffer.end() - 2)->keypoints, (dataBuffer.end() - 1)->keypoints,
                             (dataBuffer.end() - 2)->descriptors, (dataBuffer.end() - 1)->descriptors,
                             matches, descriptorType, matcherType, selectorType);*/
            matchDescriptors((dataBuffer.end() - 2)->keypoints, (dataBuffer.end() - 1)->keypoints,
                    (dataBuffer.end() - 2)->descriptors, (dataBuffer.end() - 1)->descriptors,
                    matches, match_descriptorType, matcherType, selectorType);
            keypointMatched_Num+=matches.size();
            cout<<"number of matched keypoints="<<matches.size()<<endl;

            //// EOF STUDENT ASSIGNMENT

            // store matches in current data frame
            (dataBuffer.end() - 1)->kptMatches = matches;

            cout << "#4 : MATCH KEYPOINT DESCRIPTORS done" << endl;

            // visualize matches between current and previous image
            bVis = true;
            if (bVis)
            {
                cv::Mat matchImg = ((dataBuffer.end() - 1)->cameraImg).clone();
                cv::drawMatches((dataBuffer.end() - 2)->cameraImg, (dataBuffer.end() - 2)->keypoints,
                                (dataBuffer.end() - 1)->cameraImg, (dataBuffer.end() - 1)->keypoints,
                                matches, matchImg,
                                cv::Scalar::all(-1), cv::Scalar::all(-1),
                                vector<char>(), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

                string windowName = "Matching keypoints between two camera images";
                cv::namedWindow(windowName, 7);
                cv::imshow(windowName, matchImg);
                cout << "Press key to continue to next image" << endl;
                cv::waitKey(0); // wait for key to be pressed
            }
            bVis = false;
        }

    } // eof loop over all images

    cout<<endl;
    int sum_keypoints=0;
    cout<<"keypoints number for "<<detectorType<<" detector"<<endl;
    for(auto keypointNumber_frame:keypointNumber_frames)
        {cout<<keypointNumber_frame<<",";
        sum_keypoints+=keypointNumber_frame;}
    cout<<endl;
    cout<<"total keypoints number="<<sum_keypoints<<endl;
    

    
    cout<<"keypoints "<<detectorType<<" detector"<<" Executation Time"<<endl; 
    float sum=0;
    for(auto keypointExtTime1:keypointExtTime_frames){
        cout<<keypointExtTime1<<",";
        sum+=keypointExtTime1;
    }
    cout<<endl;
    cout<<"sum="<<sum<<endl;
    //cout<<"sum="<<std::accumulate(keypointExtTime_frames.begin(),keypointExtTime_frames.end(),0);

    cout<<endl;
    cout<<"print numbers of matched keypoints"<<endl;
    cout<<detectorType<<" detector"<<endl;
    cout<<descriptorType<<" descriptor"<<endl;
    cout<<"the number of matched keypoints="<<keypointMatched_Num<<endl;
    cout<<"the total execuation time of detect plus descriptor="<<sum+DescripExtTime_Sum<<endl;

    return 0;
}
