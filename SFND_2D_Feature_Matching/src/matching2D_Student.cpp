#include <numeric>
#include "matching2D.hpp"
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp>


using namespace std;

// Find best matches for keypoints in two camera images based on several matching methods
void matchDescriptors(std::vector<cv::KeyPoint> &kPtsSource, std::vector<cv::KeyPoint> &kPtsRef, cv::Mat &descSource, cv::Mat &descRef,
                      std::vector<cv::DMatch> &matches, std::string descriptorType, std::string matcherType, std::string selectorType)
{
    // configure matcher
    bool crossCheck = false;
    cv::Ptr<cv::DescriptorMatcher> matcher;

    if (matcherType.compare("MAT_BF") == 0)
    {
        int normType;
        if(descriptorType.compare("DES_BINARY")==0)
        {
            normType = cv::NORM_HAMMING;
        }
        else
        {
            normType=cv::NORM_L2;
        }
        
        matcher = cv::BFMatcher::create(normType, crossCheck);
    }
    else if (matcherType.compare("MAT_FLANN") == 0)
    {
        // ...
        if(descSource.type()!=CV_32F||descRef.type()!=CV_32F){
            descSource.convertTo(descSource,CV_32F);
            descRef.convertTo(descRef,CV_32F);
        }
        matcher=cv::DescriptorMatcher::create(cv::DescriptorMatcher::FLANNBASED);
        cout<<"FLANN matching"<<endl;
    }

    // perform matching task
    if (selectorType.compare("SEL_NN") == 0)
    { // nearest neighbor (best match)

        matcher->match(descSource, descRef, matches); // Finds the best match for each descriptor in desc1
    }
    else if (selectorType.compare("SEL_KNN") == 0)
    { // k nearest neighbors (k=2)

        // ...
        vector<vector<cv::DMatch>> knn_matches;
        double t=(double)cv::getTickCount();
        matcher->knnMatch(descSource,descRef,knn_matches,2);
        t=((double)cv::getTickCount()-t)/cv::getTickFrequency();
        cout<<"(KNN) with n= "<<knn_matches.size()<<" matches in"<<1000*t/1.0<<" ms"<<endl;

        double minDescDistRatio=0.8;
        for(auto it=knn_matches.begin();it!=knn_matches.end();++it){
            if((*it)[0].distance<minDescDistRatio*(*it)[1].distance){
                matches.push_back((*it)[0]);
            }
        }
        cout<<"# keypoints removed="<<knn_matches.size()-matches.size()<<endl;
    }
}

// Use one of several types of state-of-art descriptors to uniquely identify keypoints
float descKeypoints(vector<cv::KeyPoint> &keypoints, cv::Mat &img, cv::Mat &descriptors, string descriptorType)
{
    // select appropriate descriptor
    cv::Ptr<cv::DescriptorExtractor> extractor;
    if (descriptorType.compare("BRISK") == 0)
    {

        int threshold = 30;        // FAST/AGAST detection threshold score.
        int octaves = 3;           // detection octaves (use 0 to do single scale)
        float patternScale = 1.0f; // apply this scale to the pattern used for sampling the neighbourhood of a keypoint.

        extractor = cv::BRISK::create(threshold, octaves, patternScale);
    }
    else
    {  if (descriptorType.compare("SIFT") == 0)
    {    extractor = cv::xfeatures2d::SiftDescriptorExtractor::create();
        //extractor=cv::SIFT:create();
        }
        if (descriptorType.compare("BRIEF") == 0){
        extractor=cv::xfeatures2d::BriefDescriptorExtractor::create();
        }
        if (descriptorType.compare("ORB") == 0){
        extractor=cv::ORB::create();
        }
        if (descriptorType.compare("FREAK") == 0){
        extractor=cv::xfeatures2d::FREAK::create();
        }
        if (descriptorType.compare("AKAZE") == 0){
        extractor=cv::AKAZE::create();
        }
        //...
    }

    // perform feature description
    double t = (double)cv::getTickCount();
    extractor->compute(img, keypoints, descriptors);
    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    cout << descriptorType << " descriptor extraction in " << 1000 * t / 1.0 << " ms" << endl;
    return 1000 * t / 1.0;
}

// Detect keypoints in image using the traditional Shi-Thomasi detector
float detKeypointsShiTomasi(vector<cv::KeyPoint> &keypoints, cv::Mat &img, bool bVis)
{
    // compute detector parameters based on image size
    int blockSize = 4;       //  size of an average block for computing a derivative covariation matrix over each pixel neighborhood
    double maxOverlap = 0.0; // max. permissible overlap between two features in %
    double minDistance = (1.0 - maxOverlap) * blockSize;
    int maxCorners = img.rows * img.cols / max(1.0, minDistance); // max. num. of keypoints

    double qualityLevel = 0.01; // minimal accepted quality of image corners
    double k = 0.04;

    // Apply corner detection
    double t = (double)cv::getTickCount();
    vector<cv::Point2f> corners;
    cv::goodFeaturesToTrack(img, corners, maxCorners, qualityLevel, minDistance, cv::Mat(), blockSize, false, k);

    // add corners to result vector
    for (auto it = corners.begin(); it != corners.end(); ++it)
    {

        cv::KeyPoint newKeyPoint;
        newKeyPoint.pt = cv::Point2f((*it).x, (*it).y);
        newKeyPoint.size = blockSize;
        keypoints.push_back(newKeyPoint);
    }
    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    cout << "Shi-Tomasi detection with n=" << keypoints.size() << " keypoints in " << 1000 * t / 1.0 << " ms" << endl;

    // visualize results
    if (bVis)
    {
        cv::Mat visImage = img.clone();
        cv::drawKeypoints(img, keypoints, visImage, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
        string windowName = "Shi-Tomasi Corner Detector Results";
        cv::namedWindow(windowName, 6);
        imshow(windowName, visImage);
        cv::waitKey(0);
    }
    return 1000*t/1.0;
}

float detKeypointsHarris(vector<cv::KeyPoint> &keypoints, cv::Mat &img, bool bVis)
{
   
    // Apply corner detection
    double t = (double)cv::getTickCount();

    // detector parameters
    int blockSize = 2;       // 
    int apertureSize=3;     //aperture parameter for Sobel operator
    int minResponse=100;
    double k = 0.04;  //Harris parameter


    // Detect Harris corners and normalize output
    cv::Mat dst,dst_norm,dst_norm_scaled;
    dst=cv::Mat::zeros(img.size(),CV_32FC1);
    cv::cornerHarris(img,dst,blockSize,apertureSize,k,cv::BORDER_DEFAULT);
    cv::normalize(dst,dst_norm,0,255,cv::NORM_MINMAX,CV_32FC1,cv::Mat());
    cv::convertScaleAbs(dst_norm,dst_norm_scaled);

    //look for prominent corners
    double maxOverlap=0.0;

    for(size_t j=0;j<dst_norm.rows;j++){
        for(size_t i=0;i<dst_norm.cols;i++)
        { int response = (int) dst_norm.at<float>(j,i);
        if(response>minResponse){
            cv::KeyPoint newKeyPoint;
            newKeyPoint.pt=cv::Point2f(i,j);
            newKeyPoint.size=2*apertureSize;
            newKeyPoint.response=response;

            //perform non-maximum suppression (NMS) in local neighbourhood around new key point
            bool bOverlap=false;
            for(auto it=keypoints.begin();it!=keypoints.end();++it)
            {
                double kptOverlap=cv::KeyPoint::overlap(newKeyPoint,*it);
                if(kptOverlap>maxOverlap){
                    bOverlap=true;
                    if(newKeyPoint.response>(*it).response){
                        *it=newKeyPoint;
                        break;
                    }
                }
            }
            if(!bOverlap){
                keypoints.push_back(newKeyPoint);
            }
        }

        } // eof loop over cols
    } // eof loop over rows

    
    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    cout << "Harris detection with n=" << keypoints.size() << " keypoints in " << 1000 * t / 1.0 << " ms" << endl;

    // visualize results
    if (bVis)
    {
        cv::Mat visImage = dst_norm_scaled.clone();
        cv::drawKeypoints(dst_norm_scaled, keypoints, visImage, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
        string windowName = "Harris Corner Detector Results";
        cv::namedWindow(windowName, 6);
        imshow(windowName, visImage);
        cv::waitKey(0);
    }
    
    return 1000*t/1.0;

}

float detKeypointsModern(std::vector<cv::KeyPoint> &keypoints, cv::Mat &img, std::string detectorType, bool bVis)
{  if (detectorType.compare("FAST") == 0)
    { int threshold=30;
      bool bNMS=true;  // perform non-maxima supperssion on keypoints
      cv::FastFeatureDetector::DetectorType type=cv::FastFeatureDetector::TYPE_9_16; //TYPE_9_16,TYPE_7_12,TYPE_5_8
      cv::Ptr<cv::FeatureDetector> detector=cv::FastFeatureDetector::create(threshold,bNMS,type);

      double t= (double) cv::getTickCount();
      detector->detect(img,keypoints);
      t=((double)cv::getTickCount()-t)/cv::getTickFrequency();
      cout<<"FAST with n="<<keypoints.size()<<"keypoints in"<<1000*t/1.0<<"ms"<<endl;

      if(bVis){
          cv::Mat visImage=img.clone();
          cv::drawKeypoints(img,keypoints,visImage,cv::Scalar::all(-1),cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
          string windowName="FAST Results";
          cv::namedWindow(windowName,2);
          imshow(windowName,visImage);
          cv::waitKey(0);
      }
    
    return 1000*t/1.0;

    }// eof FAST algorithm

    if (detectorType.compare("SIFT") == 0)
        { 
        double t= (double) cv::getTickCount();
        cv::Ptr<cv::Feature2D> sift=cv::xfeatures2d::SIFT::create();
        //cv::Ptr<cv::Feature2D> sift=cv::SIFT::create();
        sift->detect(img,keypoints);
        t=((double)cv::getTickCount()-t)/cv::getTickFrequency();
        cout<<"SIFT with n="<<keypoints.size()<<"keypoints in"<<1000*t/1.0<<"ms"<<endl;

        if(bVis){
            cv::Mat visImage=img.clone();
            cv::drawKeypoints(img,keypoints,visImage,cv::Scalar::all(-1),cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
            string windowName="SIFT Results";
            cv::namedWindow(windowName,2);
            imshow(windowName,visImage);
            cv::waitKey(0);
        }
        
        return 1000*t/1.0;

        }// eof SIFT algorithm

        if (detectorType.compare("BRISK") == 0)
        { 
        double t= (double) cv::getTickCount();
        cv::Ptr<cv::Feature2D> brisk=cv::BRISK::create();
        //cv::Ptr<cv::Feature2D> sift=SIFT::create();
        brisk->detect(img,keypoints);
        t=((double)cv::getTickCount()-t)/cv::getTickFrequency();
        cout<<"BRISK with n="<<keypoints.size()<<"keypoints in "<<1000*t/1.0<<"ms"<<endl;

        if(bVis){
            cv::Mat visImage=img.clone();
            cv::drawKeypoints(img,keypoints,visImage,cv::Scalar::all(-1),cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
            string windowName="BRISK Results";
            cv::namedWindow(windowName,2);
            imshow(windowName,visImage);
            cv::waitKey(0);
        }

        return 1000*t/1.0;

        }// eof BRISK algorithm


        if (detectorType.compare("AKAZE") == 0)
        { 
        double t= (double) cv::getTickCount();
        cv::Ptr<cv::AKAZE> akaze=cv::AKAZE::create();
        //cv::Ptr<cv::Feature2D> sift=SIFT::create();
        akaze->detect(img,keypoints);
        t=((double)cv::getTickCount()-t)/cv::getTickFrequency();
        cout<<"akaze with n="<<keypoints.size()<<" keypoints in "<<1000*t/1.0<<"ms"<<endl;

        if(bVis){
            cv::Mat visImage=img.clone();
            cv::drawKeypoints(img,keypoints,visImage,cv::Scalar::all(-1),cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
            string windowName="Akaze Results";
            cv::namedWindow(windowName,2);
            imshow(windowName,visImage);
            cv::waitKey(0);
        }
        
        return 1000*t/1.0;

        }// eof AKAZE algorithm

        
        if (detectorType.compare("ORB") == 0)
        { 
        double t= (double) cv::getTickCount();
        cv::Ptr<cv::Feature2D> orb=cv::ORB::create();
        //cv::Ptr<cv::Feature2D> sift=SIFT::create();
        orb->detect(img,keypoints);
        t=((double)cv::getTickCount()-t)/cv::getTickFrequency();
        cout<<"ORB with n="<<keypoints.size()<<" keypoints in "<<1000*t/1.0<<"ms"<<endl;

        if(bVis){
            cv::Mat visImage=img.clone();
            cv::drawKeypoints(img,keypoints,visImage,cv::Scalar::all(-1),cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
            string windowName="ORB Results";
            cv::namedWindow(windowName,2);
            imshow(windowName,visImage);
            cv::waitKey(0);
        }
        return 1000*t/1.0;

        }// eof AKAZE algorithm

}