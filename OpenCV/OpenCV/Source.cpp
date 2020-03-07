#include "opencv2/objdetect.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include <iostream>
#include <stdio.h>
#include <opencv2/core/utility.hpp>
#include <opencv2/tracking.hpp>
#include <opencv2/videoio.hpp>
#include <iostream>
#include <cstring>
#include <ctime>

using namespace std;
using namespace cv;
// OBJECT TRACKING PROTOTYPE
// Convert to string

// Declare structure to be used to pass data from C++ to Mono. FACES
struct Circle
{
	Circle(int x, int y, int radius) : X(x), Y(y), Radius(radius) {}
	int X, Y, Radius;
};

// Object Tracking Rectangles!
struct Rectangle {
    Rectangle(int width, int height, int x, int y) : Width(width), Height(height), X(x), Y(y) {}
    int Width, Height, X, Y;
};

CascadeClassifier _faceCascade;
String _windowName = "Unity OpenCV Prototype #2";
VideoCapture _capture;
int _scale = 1;
MultiTracker trackers;

extern "C" int __declspec(dllexport) __stdcall  Init(int& outCameraWidth, int& outCameraHeight)
{
	// Load LBP face cascade.
	if (!_faceCascade.load("lbpcascade_frontalface_improved.xml"))
		return -1;

	// Open the stream.
	_capture.open(0);
	if (!_capture.isOpened())
		return -2;

	outCameraWidth = _capture.get(CAP_PROP_FRAME_WIDTH);
	outCameraHeight = _capture.get(CAP_PROP_FRAME_HEIGHT);
}



// Expose the function for DLL
extern "C" void __declspec(dllexport) __stdcall  Track(Rectangle * outTracking, int maxOutTrackingCount, int& outDetectedTrackingCount)
{

    Mat frame;
    _capture >> frame;
    if(frame.empty()) {
        return;
    }

    // create the tracker


    // container of the tracked objects
    vector<Rect2d> objects;


    vector<Rect> ROIs;
    selectROIs("tracker", frame, ROIs);

    //quit when the tracked object(s) is not provided
    if (ROIs.size() < 1)
        return;

    // initialize the tracker
    std::vector<Ptr<Tracker> > algorithms;
    for (size_t i = 0; i < ROIs.size(); i++)
    {
        algorithms.push_back(cv::TrackerKCF::create());
        objects.push_back(ROIs[i]);
    }

    trackers.add(algorithms, frame, objects);

        //update the tracking result
        trackers.update(frame);

        // draw the tracked object

        for (size_t i = 0; i < trackers.getObjects().size(); i++) {
            Rect2d currentTrack = trackers.getObjects()[i];
            rectangle(frame, currentTrack, Scalar(255, 0, 0), 2, 1);
            
            outTracking[i] = Rectangle(currentTrack.width, currentTrack.height, currentTrack.x, currentTrack.y);
            outDetectedTrackingCount++;

            if (outDetectedTrackingCount == maxOutTrackingCount)
                break;

        }
        // show image with the tracked object
        imshow("tracker", frame);
    

}

extern "C" void __declspec(dllexport) __stdcall  Close()
{
	_capture.release();
}

extern "C" void __declspec(dllexport) __stdcall SetScale(int scale)
{
	_scale = scale;
}

extern "C" void __declspec(dllexport) __stdcall Detect(Circle * outFaces, int maxOutFacesCount, int& outDetectedFacesCount)
{
	Mat frame;
	// >> shifts right and adds either 0s, if value is an unsigned type, or extends the top bit (to preserve the sign) if its a signed type.
	_capture >> frame;
	if (frame.empty())
		return;

	std::vector<Rect> faces;
	// Convert the frame to grayscale for cascade detection.
	Mat grayscaleFrame;
	cvtColor(frame, grayscaleFrame, COLOR_BGR2GRAY);
	Mat resizedGray;
	// Scale down for better performance.
	resize(grayscaleFrame, resizedGray, Size(frame.cols / _scale, frame.rows / _scale));
	equalizeHist(resizedGray, resizedGray);

	// Detect faces.
	_faceCascade.detectMultiScale(resizedGray, faces);

	// Draw faces.
	for (size_t i = 0; i < faces.size(); i++)
	{
		Point center(_scale * (faces[i].x + faces[i].width / 2), _scale * (faces[i].y + faces[i].height / 2));
		ellipse(frame, center, Size(_scale * faces[i].width / 2, _scale * faces[i].height / 2), 0, 0, 360, Scalar(0, 0, 255), 4, 8, 0);

		// Send to application.
		outFaces[i] = Circle(faces[i].x, faces[i].y, faces[i].width / 2);
		outDetectedFacesCount++;

		if (outDetectedFacesCount == maxOutFacesCount)
			break;
	}

	// Display debug output.
	imshow("tracker", frame);
}