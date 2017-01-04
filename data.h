#pragma once
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>

using namespace std;
using namespace cv;

struct Data {
	vector<Mat> sources;
	vector<Vec2d> offsets;
	Mat Draw;
	vector<Mat> gradientXSources;
	vector<Mat> gradientYSources;
	int height, width; // height and width of result image.
	Mat SourceConstraints; //Matrice de contraintes de source.
	int selectSource;
    vector<Vec3b> colors;

	Data() {
		colors = vector<Vec3b>(7);
		colors[0] = Vec3b(255, 0, 0);
		colors[1] = Vec3b(0, 200, 0);
		colors[2] = Vec3b(0, 0, 255);
		colors[3] = Vec3b(0, 190, 190);
		colors[4] = Vec3b(180, 180, 0);
		colors[5] = Vec3b(128, 0, 128);
		colors[6] = Vec3b(100, 100, 100);
	}	
};

struct Argument {
	Data *D;
	int selectSource;
};

int main();
void static onMouse(int event, int x, int y, int foo, void* p);
void computePhotomontage(Data* D);