#pragma comment(lib, "Winmm.lib")
#include <iostream>
#include "opencv2/core/core_c.h"
#include "opencv2/core/core.hpp"
#include "opencv2/flann/miniflann.hpp"
#include "opencv2/imgproc/imgproc_c.h"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/video/video.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/ml/ml.hpp"
//#include "opencv2/highgui/highgui_c.h"
//#include "opencv2/highgui/highgui.hpp"
#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <vector>
#include <windows.h>
#include <mmsystem.h>

using namespace std;
using namespace cv;

int main()
{
    const int MAX_FEATURES = 1000;
    const int hImg = 500;
    const int wImg = 520;

    bool detection = false;
    int frameCounter = 0;

    vector<KeyPoint> kp1, kp2;
    Mat des1, des2, imgWebCam, imgFeatures;

    VideoCapture cap = VideoCapture(0, CAP_DSHOW);
    Mat vidFrame;

    Mat imgTarget = imread("ddario.jpeg");
    VideoCapture myVid = VideoCapture("dram.mp4");

    resize(imgTarget, imgTarget, Size(wImg, hImg));

    myVid >> vidFrame;
    resize(vidFrame, vidFrame, Size(wImg, hImg));

    Ptr<ORB> orb = ORB::create(MAX_FEATURES);
    orb->detectAndCompute(imgTarget, Mat(), kp1, des1);
    //drawKeypoints(imgTarget, kp1, imgTarget, Scalar::all(-1), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

    while(1)
    {
        cap >> imgWebCam;
        Mat imgAug = imgWebCam.clone();
        orb->detectAndCompute(imgWebCam, Mat(), kp2, des2);
        //drawKeypoints(imgWebCam, kp2, imgWebCam, Scalar::all(-1), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
        
        if (!detection) {
            myVid.set(CAP_PROP_POS_FRAMES, 0);
            frameCounter = 0;
            PlaySound(NULL, 0, 0);
        }
        else {
            if (frameCounter == myVid.get(CAP_PROP_FRAME_COUNT)) {
                myVid.set(CAP_PROP_POS_FRAMES, 0);
                frameCounter = 0;
                PlaySound(NULL, 0, 0);
            }
            else if (frameCounter == 1)
                PlaySound(L"nsksi.wav", NULL, SND_FILENAME | SND_ASYNC);
            myVid >> vidFrame;
            resize(vidFrame, vidFrame, Size(wImg, hImg));
        }

        BFMatcher matcher(NORM_L2);
        vector<vector<DMatch>> matches;
        if (!des1.empty() && !des2.empty())
            matcher.knnMatch(des1, des2, matches, 2);
        vector<DMatch> good = {};
        int i = 0;
        while (i < matches.size()) {
            if (matches[i][0].distance < 0.75 * matches[i][1].distance)
                good.push_back(matches[i][0]);
            i++;
        }
        if (!good.empty())
            cout << good.size() << endl;
        drawMatches(imgTarget, kp1, imgWebCam, kp2, good, imgFeatures);

        vector<Point2f> srcPts;
        vector<Point2f> dstPts;

        if (good.size() > 20) {
            detection = true;
            for (size_t i = 0; i < good.size(); i++) {
                srcPts.push_back(kp1[good[i].queryIdx].pt);
                //srcPts[i].reshape(-1, {1, 2});
                dstPts.push_back(kp2[good[i].trainIdx].pt);
            }
            Mat matrix = findHomography(srcPts, dstPts, RANSAC, 5);
            cout << matrix << endl;

            vector<Point2f> pts(4);
            pts[0] = Point2f(0, 0);
            pts[1] = Point2f((float)imgTarget.cols, 0);
            pts[2] = Point2f((float)imgTarget.cols, (float)imgTarget.rows);
            pts[3] = Point2f(0, (float)imgTarget.rows);
            vector<Point2f> dst(4);
            perspectiveTransform(pts, dst, matrix);
            line(imgFeatures, dst[0] + Point2f((float)imgTarget.cols, 0),
                dst[1] + Point2f((float)imgTarget.cols, 0), Scalar(255, 0, 255), 4);
            line(imgFeatures, dst[1] + Point2f((float)imgTarget.cols, 0),
                dst[2] + Point2f((float)imgTarget.cols, 0), Scalar(255, 0, 255), 4);
            line(imgFeatures, dst[2] + Point2f((float)imgTarget.cols, 0),
                dst[3] + Point2f((float)imgTarget.cols, 0), Scalar(255, 0, 255), 4);
            line(imgFeatures, dst[3] + Point2f((float)imgTarget.cols, 0),
                dst[0] + Point2f((float)imgTarget.cols, 0), Scalar(255, 0, 255), 4);

            Mat imgWarp;
            warpPerspective(vidFrame, imgWarp, matrix, imgWebCam.size());

            Mat maskNew = Mat::zeros(imgWebCam.rows, imgWebCam.cols, 0);
            vector<Point> pt;
            for (int j = 0; j < dst.size(); j++) {
                pt.push_back(dst[j]);
            }
            fillPoly(maskNew, pt, Scalar(255, 255, 255));

            Mat maskInv;
            bitwise_not(maskNew, maskInv);
            Mat tempAug;
            bitwise_and(imgAug, imgAug, tempAug, maskInv);
            bitwise_or(tempAug, imgWarp, imgAug);

            imshow("imgAug", imgAug);
            //imshow("maskNew", maskInv);
            //imshow("imgWarp", imgWarp);
        }
        else
            detection = false;
        
        frameCounter++;
        //imshow("imgFeatures", imgFeatures);
        imshow("Live", imgWebCam);
        //imshow("imgTarget", imgTarget);
        //imshow("vidFrame", vidFrame);
        if (waitKey(1) >= 0)
            break;
    }
    cap.release();
    destroyAllWindows();
}
