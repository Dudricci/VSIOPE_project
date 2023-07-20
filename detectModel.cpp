#include "opencv2/objdetect.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/videoio.hpp"
#include <opencv2/core/utility.hpp>

#include <iostream>
#include <chrono>
#include <algorithm>
#include <iterator>
#include <random>
#include <string>
using namespace std;
using namespace cv;
using namespace std::chrono;

void detectAndDisplay( Mat frame );
CascadeClassifier bottle_cascade;
int main( int argc, const char** argv )
{
    CommandLineParser parser(argc, argv,
                             "{help h||}"
                             "{bottle_cascade|../../classifierCPP/cascade/cascade.xml|Path to bottle cascade.}"
                             "{foto|../upper_275176.jpg|foto path}"
                             "{camera|0|Camera device number.}");
    parser.about( "\nThis program demonstrates using the cv::CascadeClassifier class to detect bottle in a video stream.\n"
                  "You can use Haar or LBP features.\n\n" );
    parser.printMessage();
    String bottle_cascade_name =  parser.get<String>("bottle_cascade") ;
    //-- 1. Load the cascades
    if( !bottle_cascade.load( bottle_cascade_name ) )
    {
        cout << "--(!)Error loading bottle cascade\n";
        return -1;
    };

    //Mat foto = parser.get<Mat>("foto");

    int camera_device = parser.get<int>("camera");
    VideoCapture capture;
    //-- 2. Read the video stream
    capture.open( camera_device );
    if ( ! capture.isOpened() )
    {
        cout << "--(!)Error opening video capture\n";
        return -1;
    }
    Mat frame;
    int z=0;
    while ( capture.read(frame) )
    {
        if( frame.empty() )
        {
            cout << "--(!) No captured frame -- Break!\n";
            break;
        }
        //-- 3. Apply the classifier to the frame
        if(z<=30){
            uint64_t ms = duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count();
            //std::cout << ms << " milliseconds since the Epoch\n";

            detectAndDisplay( frame );

            uint64_t sec = duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count();
            std::cout << sec-ms << " millisecodns of inference\n";
        }
        else{
            detectAndDisplay( frame );
        }
        z=z+1;
        //detectAndDisplay( frame );
        if( waitKey(10) == 27 )
        {
            break; // escape
        }
    }
    return 0;
}

void bottleDetection(Mat frame)
{
    Mat frame_gray;
    cvtColor( frame, frame_gray, COLOR_BGR2GRAY );
    equalizeHist( frame_gray, frame_gray );
    //-- Detect bottle
    std::vector<Rect> bottle;
    bottle_cascade.detectMultiScale( frame_gray, bottle );
    for ( size_t i = 0; i < bottle.size(); i++ )
    {
        Point center( bottle[i].x + bottle[i].width/2, bottle[i].y + bottle[i].height/2 );
        rectangle(frame, Rect(center.x - bottle[i].width/2, center.y - bottle[i].height/2, bottle[i].width, bottle[i].height), Scalar(255, 0, 255), 4);
        Mat faceROI = frame_gray( bottle[i] );
    }
    //-- Show what you got
    imshow( "Capture - Face detection", frame );
}

void detectAndDisplay( Mat frame )
{
    Mat frame_gray;
    cvtColor( frame, frame_gray, COLOR_BGR2GRAY );
    equalizeHist( frame_gray, frame_gray );
    //-- Detect bottle
    std::vector<Rect> bottle;
    bottle_cascade.detectMultiScale( frame_gray, bottle , 1.5, 2, 0 | CV_HAAR_SCALE_IMAGE, cv::Size(30, 40));
    for ( size_t i = 0; i < bottle.size(); i++ )
    {
        Point center( bottle[i].x + bottle[i].width/2, bottle[i].y + bottle[i].height/2 );
        rectangle(frame, Rect(center.x - bottle[i].width/2, center.y - bottle[i].height/2, bottle[i].width, bottle[i].height), Scalar(255, 0, 255), 4);
        Mat faceROI = frame_gray( bottle[i] );
    }
    //-- Show what you got
    imshow( "Capture - Bottle detection", frame );
}
