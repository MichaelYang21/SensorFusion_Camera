
#include <iostream>
#include <algorithm>
#include <numeric>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "camFusion.hpp"
#include "dataStructures.h"

using namespace std;
using namespace cv;


// Create groups of Lidar points whose projection into the camera falls into the same bounding box
void clusterLidarWithROI(std::vector<BoundingBox> &boundingBoxes, std::vector<LidarPoint> &lidarPoints, float shrinkFactor, cv::Mat &P_rect_xx, cv::Mat &R_rect_xx, cv::Mat &RT)
{
    // loop over all Lidar points and associate them to a 2D bounding box
    cv::Mat X(4, 1, cv::DataType<double>::type);
    cv::Mat Y(3, 1, cv::DataType<double>::type);

    for (auto it1 = lidarPoints.begin(); it1 != lidarPoints.end(); ++it1)
    {
        // assemble vector for matrix-vector-multiplication
        X.at<double>(0, 0) = it1->x;
        X.at<double>(1, 0) = it1->y;
        X.at<double>(2, 0) = it1->z;
        X.at<double>(3, 0) = 1;

        // project Lidar point into camera
        Y = P_rect_xx * R_rect_xx * RT * X;
        cv::Point pt;
        // pixel coordinates
        pt.x = Y.at<double>(0, 0) / Y.at<double>(2, 0); 
        pt.y = Y.at<double>(1, 0) / Y.at<double>(2, 0); 

        vector<vector<BoundingBox>::iterator> enclosingBoxes; // pointers to all bounding boxes which enclose the current Lidar point
        for (vector<BoundingBox>::iterator it2 = boundingBoxes.begin(); it2 != boundingBoxes.end(); ++it2)
        {
            // shrink current bounding box slightly to avoid having too many outlier points around the edges
            cv::Rect smallerBox;
            smallerBox.x = (*it2).roi.x + shrinkFactor * (*it2).roi.width / 2.0;
            smallerBox.y = (*it2).roi.y + shrinkFactor * (*it2).roi.height / 2.0;
            smallerBox.width = (*it2).roi.width * (1 - shrinkFactor);
            smallerBox.height = (*it2).roi.height * (1 - shrinkFactor);

            // check wether point is within current bounding box
            if (smallerBox.contains(pt))
            {
                enclosingBoxes.push_back(it2);
            }

        } // eof loop over all bounding boxes

        // check wether point has been enclosed by one or by multiple boxes
        if (enclosingBoxes.size() == 1)
        { 
            // add Lidar point to bounding box
            enclosingBoxes[0]->lidarPoints.push_back(*it1);
        }

    } // eof loop over all Lidar points
}

/* 
* The show3DObjects() function below can handle different output image sizes, but the text output has been manually tuned to fit the 2000x2000 size. 
* However, you can make this function work for other sizes too.
* For instance, to use a 1000x1000 size, adjusting the text positions by dividing them by 2.
*/
void show3DObjects(std::vector<BoundingBox> &boundingBoxes, cv::Size worldSize, cv::Size imageSize, bool bWait)
{
    // create topview image
    cv::Mat topviewImg(imageSize, CV_8UC3, cv::Scalar(255, 255, 255));

    for(auto it1=boundingBoxes.begin(); it1!=boundingBoxes.end(); ++it1)
    {
        // create randomized color for current 3D object
        cv::RNG rng(it1->boxID);
        cv::Scalar currColor = cv::Scalar(rng.uniform(0,150), rng.uniform(0, 150), rng.uniform(0, 150));

        // plot Lidar points into top view image
        int top=1e8, left=1e8, bottom=0.0, right=0.0; 
        float xwmin=1e8, ywmin=1e8, ywmax=-1e8;
        for (auto it2 = it1->lidarPoints.begin(); it2 != it1->lidarPoints.end(); ++it2)
        {
            // world coordinates
            float xw = (*it2).x; // world position in m with x facing forward from sensor
            float yw = (*it2).y; // world position in m with y facing left from sensor
            xwmin = xwmin<xw ? xwmin : xw;
            ywmin = ywmin<yw ? ywmin : yw;
            ywmax = ywmax>yw ? ywmax : yw;

            // top-view coordinates
            int y = (-xw * imageSize.height / worldSize.height) + imageSize.height;
            int x = (-yw * imageSize.width / worldSize.width) + imageSize.width / 2;

            // find enclosing rectangle
            top = top<y ? top : y;
            left = left<x ? left : x;
            bottom = bottom>y ? bottom : y;
            right = right>x ? right : x;

            // draw individual point
            cv::circle(topviewImg, cv::Point(x, y), 4, currColor, -1);
        }

        // draw enclosing rectangle
        cv::rectangle(topviewImg, cv::Point(left, top), cv::Point(right, bottom),cv::Scalar(0,0,0), 2);

        // augment object with some key data
        char str1[200], str2[200];
        sprintf(str1, "id=%d, #pts=%d", it1->boxID, (int)it1->lidarPoints.size());
        putText(topviewImg, str1, cv::Point2f(left-250, bottom+50), cv::FONT_ITALIC, 2, currColor);
        sprintf(str2, "xmin=%2.2f m, yw=%2.2f m", xwmin, ywmax-ywmin);
        putText(topviewImg, str2, cv::Point2f(left-250, bottom+125), cv::FONT_ITALIC, 2, currColor);  
    }

    // plot distance markers
    float lineSpacing = 2.0; // gap between distance markers
    int nMarkers = floor(worldSize.height / lineSpacing);
    for (size_t i = 0; i < nMarkers; ++i)
    {
        int y = (-(i * lineSpacing) * imageSize.height / worldSize.height) + imageSize.height;
        cv::line(topviewImg, cv::Point(0, y), cv::Point(imageSize.width, y), cv::Scalar(255, 0, 0));
    }

    // display image
    string windowName = "3D Objects";
    cv::namedWindow(windowName, 1);
    cv::imshow(windowName, topviewImg);

    if(bWait)
    {
        cv::waitKey(0); // wait for key to be pressed
    }
}


// associate a given bounding box with the keypoints it contains
void clusterKptMatchesWithROI(BoundingBox &boundingBox, std::vector<cv::KeyPoint> &kptsPrev, std::vector<cv::KeyPoint> &kptsCurr, std::vector<cv::DMatch> &kptMatches)
{
    // ...
    int kptCurrIdx,kptPrevIdx;
    vector<cv::DMatch> MatchesBBox;
    vector<double> distance;
    for(auto it=kptMatches.begin();it!=kptMatches.end();++it){
        kptCurrIdx=(*it).trainIdx;
        kptPrevIdx=(*it).queryIdx;
        if(boundingBox.roi.contains(kptsCurr[kptCurrIdx].pt)){
            MatchesBBox.push_back((*it));
            distance.push_back(cv::norm(kptsCurr[kptCurrIdx].pt-kptsPrev[kptPrevIdx].pt));
        }
    }

    cout<<"size of keypoints in ROI "<<MatchesBBox.size()<<endl;

    double distance_mean=0;
    for(auto it=distance.begin();it!=distance.end();++it){
        distance_mean+=*it;
    }
    distance_mean=distance_mean/distance.size();

    double distance_threshold=0.8*distance_mean;
    for(int i=0;i<distance.size();i++){
        if(distance[i]<distance_threshold){
            boundingBox.kptMatches.push_back(MatchesBBox[i]);
            boundingBox.keypoints.push_back(kptsCurr[MatchesBBox[i].trainIdx]);
        }
    }

    cout<<"size of keypoints in ROI "<<boundingBox.kptMatches.size()<<" after outliers are removed"<<endl;


}


// Compute time-to-collision (TTC) based on keypoint correspondences in successive images
void computeTTCCamera(std::vector<cv::KeyPoint> &kptsPrev, std::vector<cv::KeyPoint> &kptsCurr, 
                      std::vector<cv::DMatch> kptMatches, double frameRate, double &TTC, cv::Mat *visImg)
{
    // ...
    cout<<"assemble vectors of distRatios"<<endl;
    vector<double> distRatios;
    for(auto it1=kptMatches.begin();it1!=kptMatches.end()-1;++it1){
        KeyPoint kpOuterCurr=kptsCurr.at(it1->trainIdx);
        KeyPoint kpOuterPrev=kptsPrev.at(it1->queryIdx);

        for(auto it2=kptMatches.begin()+1;it2!=kptMatches.end();++it2){

            double minDist=100.0;

            KeyPoint kpInnerCurr=kptsCurr.at(it2->trainIdx);
            KeyPoint kpInnerPrev=kptsPrev.at(it2->queryIdx);

            double distCurr=cv::norm(kpOuterCurr.pt-kpInnerCurr.pt);
            double distPrev=cv::norm(kpOuterPrev.pt-kpInnerPrev.pt);

            if(distPrev> std::numeric_limits<double>::epsilon() && distCurr>=minDist){
                double distRatio=distCurr/distPrev;
                distRatios.push_back(distRatio);
        }
        
    }
    }

    if(distRatios.size()==0)
    { TTC=NAN;
       return;
    }
    cout<<"calculating median values of distance ratio"<<endl;
    sort(distRatios.begin(),distRatios.end());
    long medIndex=floor(distRatios.size()/2.0);
    double medDistRatio=distRatios.size()%2==0?(distRatios[medIndex-1]+distRatios[medIndex])/2.0:distRatios[medIndex];

    double dT=1/frameRate;
    TTC=-dT/(1-medDistRatio);
}


void computeTTCLidar(std::vector<LidarPoint> &lidarPointsPrev,
                     std::vector<LidarPoint> &lidarPointsCurr, double frameRate, double &TTC)
{
    // ...
    double dT=1.0/frameRate;
    double laneWidth=4.0;

    double minXPrev=1e9, minXCurr=1e9;
    for(auto it=lidarPointsPrev.begin();it!=lidarPointsPrev.end();++it){
        if(abs(it->y)<=laneWidth/2.0){
            minXPrev=minXPrev>it->x?it->x:minXPrev;
        }
    }

    for(auto it=lidarPointsCurr.begin();it!=lidarPointsCurr.end();++it){
        if(abs(it->y)<=laneWidth/2.0){
            minXCurr=minXCurr>it->x?it->x:minXCurr;
        }
    }
    TTC=minXCurr*dT/(minXPrev-minXCurr);
}


void matchBoundingBoxes(std::vector<cv::DMatch> &matches, std::map<int, int> &bbBestMatches, DataFrame &prevFrame, DataFrame &currFrame)
{
    // ...
    int prev_kpt_idx,curr_kpt_idx;
    int prev_BBox_size=prevFrame.boundingBoxes.size();
    int curr_BBox_size=currFrame.boundingBoxes.size();
    int counts[prev_BBox_size][curr_BBox_size]={};

    vector<int> prev_BBox_Ids,curr_BBox_Ids;
    cv::KeyPoint prev_kpt,curr_kpt;

    for(auto it=matches.begin();it!=matches.end();++it){
       prev_kpt_idx=(*it).queryIdx;
       curr_kpt_idx=(*it).trainIdx;
       prev_kpt=prevFrame.keypoints[prev_kpt_idx];
       curr_kpt=currFrame.keypoints[curr_kpt_idx];

       prev_BBox_Ids.clear();
       curr_BBox_Ids.clear();

       for(auto it2=prevFrame.boundingBoxes.begin();it2!=prevFrame.boundingBoxes.end();++it2){
           if((*it2).roi.contains(prev_kpt.pt)){
               prev_BBox_Ids.push_back((*it2).boxID);}
       }

        for(auto it3=currFrame.boundingBoxes.begin();it3!=currFrame.boundingBoxes.end();++it3){
                if((*it3).roi.contains(curr_kpt.pt)){
                    curr_BBox_Ids.push_back((*it3).boxID);}
            }
        
        for(auto prevId:prev_BBox_Ids){
            for(auto currId:curr_BBox_Ids){
                counts[prevId][currId]++;
            }
        }
    }

    int Count_Max=0;
    int Id_max;

    for(int prevId=0;prevId<prev_BBox_size;prevId++){
        Count_Max=0;
        for(int currId=0;currId<curr_BBox_size;currId++){
            if(counts[prevId][currId]>Count_Max){
                Count_Max=counts[prevId][currId];
                Id_max=currId;
            }
        }
        bbBestMatches[prevId]=Id_max;
    }


}
