#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>

#include "maxflow/graph.h"
#include "collage.h"
#include "data.h"

using namespace std;
using namespace cv;

const String winName = "Image";
void static computeGradient(const Mat &I, Mat *Gx, Mat *Gy);

void computePhotomontage(Data* D) {
	Collage C(D);
	cout << "Computing photo montage...";
	C.computePhotomontage(DESIGNATED_SOURCE, COLORS_AND_GRADIENTS);
	cout << "Done.";
}

void static onMouse(int event, int x, int y, int foo, void* p) {
	Data* D = (Data*)p;
	if (foo == CV_EVENT_FLAG_LBUTTON + CV_EVENT_MOUSEMOVE) {
		Point p0(x, y);
		circle(D->Draw, p0, 2, D->colors[D->selectSource], 2);
		circle(D->SourceConstraints, p0, 2, D->selectSource, 2);
		imshow(winName, D->Draw);
	}
	else if (event == CV_EVENT_RBUTTONDOWN) {
		D->selectSource++;
		if (D->selectSource < D->sources.size()) {
			Vec2d offset = D->offsets[D->selectSource];
			D->Draw = D->sources[0].clone();
			for (int i = 0; i < D->height; i++) {
				for (int j = 0; j < D->width; j++) {
					if (Collage::isInImage(i, j, offset[0], offset[1], D->sources[D->selectSource])) {
						D->Draw.at<Vec3b>(i, j) = D->sources[D->selectSource].at<Vec3b>(i - offset[0], j - offset[1]);
					}
					if (D->SourceConstraints.at<uchar>(i,j) != 255) {
						D->Draw.at<Vec3b>(i, j) = D->colors[D->SourceConstraints.at<uchar>(i, j)];
					}
				}
			}
			imshow(winName, D->Draw);
		}
		else {
			setMouseCallback(winName, NULL, NULL);
			computePhotomontage(D);
			cout << "Photomontage computed." << endl;
		}
	}
}

int main() {

	namedWindow(winName);
	Data D;
	int N = 4;
	D.sources = vector<Mat>(N);
	D.gradientYSources = vector<Mat>(N);
	D.gradientXSources = vector<Mat>(N);
	D.offsets = vector<Vec2d>(N);

	/*int N = 3;
	D.sources[0] = imread("../riviere.jpg");
	D.sources[1] = imread("../maison.jpg");
	D.sources[2] = imread("../cascade.jpg");
	*/

	D.sources[0] = imread("images/famille1.jpg");
	D.sources[1] = imread("images/famille2.jpg");
	D.sources[2] = imread("images/famille3.jpg");
	D.sources[3] = imread("images/famille4.jpg");

	/* Compute gradient images for all sources */
	for (int i = 0; i < N; i ++) {
		Mat I = D.sources[i];
		int m = I.rows, n = I.cols;
		Mat Gx(m, n, CV_64FC3);
		Mat Gy(m, n, CV_64FC3);
		computeGradient(I, &Gx, &Gy);
		D.gradientXSources[i] = Gx;
		D.gradientYSources[i] = Gy;
	}

		/* Define offsets for all sources */
	for (int i = 0; i < N; i++) {
		Vec2d offset = Vec2d(0, 0);
		D.offsets[i] = offset;
	}


	/* Define result image and constraints associated */
	D.height = D.sources[0].rows;
	D.width = D.sources[0].cols;
	D.SourceConstraints = Mat(D.height, D.width, CV_8UC1);
	for (int i = 0; i < D.height; i++) {
		for (int j = 0; j < D.width; j++) {
			D.SourceConstraints.at<uchar>(i, j) = 255;
		}
	}

	D.selectSource = 0;
	D.Draw = D.sources[0].clone();
	imshow(winName, D.Draw);
	setMouseCallback(winName, onMouse, &D);
	waitKey();
	return 0;
}

void static computeGradient(const Mat &I, Mat *Gx, Mat *Gy) {
	int m = I.rows, n = I.cols;
	for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            Vec3d gx, gy;
            if (i == 0 || i == m-1) {
            	gy = Vec3d(0, 0, 0);
            } else {
            	gy = ((Vec3d)I.at<Vec3b>(i + 1, j) - (Vec3d)I.at<Vec3b>(i - 1, j)) / 2;
            }

            if (j == 0 || j == n - 1) {
            	gx = Vec3d(0, 0, 0);
            } else {
            	gx = ((Vec3d)I.at<Vec3b>(i,j+1)- (Vec3d)I.at<Vec3b>(i,j-1)) / 2;
            }
            Gx->at<Vec3d>(i, j) = gx;
            Gy->at<Vec3d>(i, j) = gy;
        }
    }
}
