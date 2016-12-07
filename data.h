#pragma once
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>

using namespace std;
using namespace cv;

struct Data {
	vector<Mat> sources;
	Mat Draw;
	vector<Mat> gradientXSources;
	vector<Mat> gradientYSources;
	int height, width;
	Mat SourceConstraints; //Matrice de contraintes de source
	int selectSource;	
};

struct Argument {
	Data *D;
	int selectSource;
};

int main();
void static onMouse(int event, int x, int y, int foo, void* p);
void computePhotomontage(Data* D);