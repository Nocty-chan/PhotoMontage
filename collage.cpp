#include "collage.h"

#include <math.h>

double dataPenalty(int alpha, int label, Data *D, int i, int j, INSIDE_MODE insideMode); 
double interactionPenalty(int pi, int pj, int qi, int qj, Data *D, int labelP, int labelQ, OUTSIDE_MODE outsideMode);
void computeGraph(Graph<double, double, double> &G, INSIDE_MODE insideMode, OUTSIDE_MODE outsideMode, const Mat &R0, Data *D, int alpha);

Collage::Collage(Data* Dat) {
	Collage::D = Dat;
}

int Collage::getNImages() {
	return nImages;
}

int Collage::getImageHeight() {
	return D->height;
}

int Collage::getImageWidth() {
	return D->width;
}
 
//alpha_expansion avec plus de deux sources
void Collage::computePhotomontage(INSIDE_MODE insideMode, OUTSIDE_MODE outsideMode) {

	Mat R0(D->height, D->width, CV_8UC1);
	for (int i = 0; i < D->height; i++) {
		for (int j = 0; j < D->width; j++) {
			R0.at<uchar>(i, j) = 0;
		}
	}

	int minCut = INT_MAX;
	bool amelioration, skipZero = true;

	do {
		amelioration = false;
		for (int alpha = 0; alpha < D->sources.size(); alpha++) {
			if (skipZero && alpha == 0) {
				skipZero = false;
				continue;
			}

			Graph<double, double, double> G(3 * D->height * D->width - D->width - D->height, 2 * D->height * D->width - D->width - D->height);
			computeGraph(G, insideMode, outsideMode, R0, D, alpha);

			int flow = G.maxflow();
			if (flow < minCut) {
				amelioration = true;
				minCut = flow;
			}

			for (int i = 0; i < D->height; i++) {
				for (int j = 0; j < D->width; j++) {
					if (G.what_segment(D->width * i + j) == Graph<double, double, double>::SINK) {
						R0.at<uchar>(i, j) = alpha;
					}
				}
			}
		}

	} while (amelioration);

	Mat R(D->height, D->width, CV_8UC3);
	for (int i = 0; i < D->height; i++) {
		for (int j = 0; j < D->width; j++) {
			for (int k = 0; k < D->sources.size(); k++) {
				if (R0.at<uchar>(i, j) == k) {
					R.at<Vec3b>(i, j) = D->sources[k].at<Vec3b>(i, j);
					break;
				}
			}

		}
	}

	imshow("Photomontage", R);
	cv::waitKey();
}


double dataPenalty(int alpha, int label, Data *D, int i, int j, INSIDE_MODE insideMode) {
  double imageConstraint = D->SourceConstraints.at<uchar>(i, j);
  switch(insideMode) {
  	case DESIGNATED_SOURCE:
  		if (label != alpha) {
  			if (imageConstraint == 255 || imageConstraint == label) return 0;
  			} else {
  				if (imageConstraint == alpha || imageConstraint == 255) return 0;
  			}
  		return INT_MAX;
  }
}

double interactionPenalty(int pi, int pj, int qi, int qj, Data *D, int labelP, int labelQ, OUTSIDE_MODE outsideMode) {
  Mat SLP = D->sources[labelP];
  Mat SLQ = D->sources[labelQ];
  Mat GXLP = D->gradientXSources[labelP];
  Mat GYLP = D->gradientYSources[labelP];
  Mat GXLQ = D->gradientXSources[labelQ];
  Mat GYLQ = D->gradientYSources[labelQ];
  double termP, termQ, termPx, termPy, termQx, termQy;

  switch(outsideMode) {
  	case COLORS:
  		termP = norm((Vec3d)SLP.at<Vec3b>(pi, pj)-(Vec3d)SLQ.at<Vec3b>(pi, pj));
  		termQ = norm((Vec3d)SLP.at<Vec3b>(qi, qj)-(Vec3d)SLQ.at<Vec3b>(qi, qj));
  		return termP + termQ;
  	case GRADIENTS:
  		termPx = norm(GXLP.at<Vec3d>(pi, pj)-GXLQ.at<Vec3d>(pi, pj));
  		termPy = norm(GYLP.at<Vec3d>(pi, pj)-GYLQ.at<Vec3d>(pi, pj));
  		termQx = norm(GXLP.at<Vec3d>(qi, qj)-GXLQ.at<Vec3d>(qi, qj));  	
  		termQy = norm(GYLP.at<Vec3d>(qi, qj)-GYLQ.at<Vec3d>(qi, qj));
  		return sqrt(termPx * termPx + termPy * termPy)  + sqrt(termQx * termQx + termQy * termQy);
  	case COLORS_AND_GRADIENTS:
  		termP = norm((Vec3d)SLP.at<Vec3b>(pi, pj)-(Vec3d)SLQ.at<Vec3b>(pi, pj));
  		termQ = norm((Vec3d)SLP.at<Vec3b>(qi, qj)-(Vec3d)SLQ.at<Vec3b>(qi, qj));
  		termPx = norm(GXLP.at<Vec3d>(pi, pj)-GXLQ.at<Vec3d>(pi, pj));
  		termPy = norm(GYLP.at<Vec3d>(pi, pj)-GYLQ.at<Vec3d>(pi, pj));
  		termQx = norm(GXLP.at<Vec3d>(qi, qj)-GXLQ.at<Vec3d>(qi, qj));  	
  		termQy = norm(GYLP.at<Vec3d>(qi, qj)-GYLQ.at<Vec3d>(qi, qj));
  		return termP + termQ + sqrt(termPx * termPx + termPy * termPy)  + sqrt(termQx * termQx + termQy * termQy);
  }	
}

void computeGraph(Graph<double, double, double> &G, INSIDE_MODE insideMode, OUTSIDE_MODE outsideMode, const Mat &R0, Data *D, int alpha) {
	G.add_node(D->height * D->width); //les premiers noeuds seront les pixels, les suivants les voisinages entre deux pixels
	int middleNode = D->height * D->width;

	for (int i = 0; i < D->height; i++) {
		for (int j = 0; j < D->width; j++) {
			double currentImage = R0.at<uchar>(i, j);
			double capacityToPuits = dataPenalty(alpha, currentImage, D, i, j, insideMode);
			double capacityToSource = dataPenalty(alpha, alpha, D, i, j, insideMode);

			G.add_tweights(D->width*i + j, capacityToSource, capacityToPuits);

			if (i < D->height - 1) {
				int currVoisin = R0.at<uchar>(i + 1, j);
				G.add_node(1);
				capacityToPuits = interactionPenalty(i, j, i + 1, j, D, currentImage, currVoisin, outsideMode);
				G.add_tweights(middleNode, 0, capacityToPuits);
				double capacityToP = interactionPenalty(i, j, i + 1, j, D, currentImage, alpha, outsideMode);
				double capacityToQ = interactionPenalty(i, j, i + 1, j, D, alpha, currVoisin, outsideMode);
				G.add_edge(D->width * i + j, middleNode, capacityToP, capacityToP);
				G.add_edge(D->width * (i + 1) + j, middleNode, capacityToQ, capacityToQ);
				middleNode++;
			}

			if (j < D->width - 1) {
				int currVoisin = R0.at<uchar>(i, j + 1);
				G.add_node(1);
				capacityToPuits = interactionPenalty(i, j, i, j + 1, D, currentImage, currVoisin, outsideMode);
				if (currentImage != currVoisin) capacityToPuits++;
				G.add_tweights(middleNode, 0, capacityToPuits);
				double capacityToP = interactionPenalty(i, j, i, j + 1, D, currentImage, alpha, outsideMode);
				double capacityToQ = interactionPenalty(i, j, i, j + 1, D, alpha, currVoisin, outsideMode);
				G.add_edge(D->width * i + j, middleNode, capacityToP, capacityToP);
				G.add_edge(D->width * i + j + 1, middleNode, capacityToQ, capacityToQ);
				middleNode++;
			}
		}
	}
}
