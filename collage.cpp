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
	//cout << "Computing photomontage" << endl;
	/* Labels to be put on result image */
	Mat resultLabel(D->height, D->width, CV_8UC1);
	for (int i = 0; i < D->height; i++) {
		for (int j = 0; j < D->width; j++) {
			resultLabel.at<uchar>(i, j) = 0;
		}
	}

	int minCut = INT_MAX;
	bool amelioration, skipZero = true;
	/* Alpha expansion */ 
	do {
		amelioration = false;
		for (int alpha = 0; alpha < D->sources.size(); alpha++) {
			if (skipZero && alpha == 0) {
				skipZero = false;
				continue;
			}

			Graph<double, double, double> G(
				3 * D->height * D->width - D->width - D->height,
				2 * D->height * D->width - D->width - D->height);
			//cout << "Computing graph..." << endl;
			computeGraph(G, insideMode, outsideMode, resultLabel, D, alpha);
			//cout << "Graph computed" << endl;
			int flow = G.maxflow();
			if (flow < minCut) {
				amelioration = true;
				minCut = flow;
			}

			for (int i = 0; i < D->height; i++) {
				for (int j = 0; j < D->width; j++) {
					if (G.what_segment(D->width * i + j) == Graph<double, double, double>::SINK) {
							resultLabel.at<uchar>(i, j) = alpha;
					}
				}
			}
		}

	} while (amelioration);
	
	/* Compute result image */ 
	Mat resultImage(D->height, D->width, CV_8UC3);
	for (int i = 0; i < D->height; i++) {
		for (int j = 0; j < D->width; j++) {
			for (int k = 0; k < D->sources.size(); k++) {
				Vec2d offset = D->offsets[k];
				if (resultLabel.at<uchar>(i, j) == k) {
					resultImage.at<Vec3b>(i, j) = D->sources[k].at<Vec3b>(i - offset[0], j - offset[1]);
					if (D->SourceConstraints.at<uchar>(i, j) == 255) {
						D->Draw.at<Vec3b>(i, j) = (D->sources[k].at<Vec3b>(i - offset[0], j - offset[1]) + D->colors[k % 7]) / 2;
					} else {
						D->Draw.at<Vec3b>(i, j) = D->colors[k % 7];
					}
					break;
				}
			}
		}
	}

	imshow("Image", D->Draw);
	imshow("Photomontage", resultImage);
	cv::waitKey();
}

/* calcule la pénalité pour un pixel */
double dataPenalty(int alpha, int label, Data *D, int i, int j, INSIDE_MODE insideMode) {
	//cout << "computing dataPenalty" << endl;
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

/* calcule la pénalité d'interaction entre deux pixels P et Q */ 
double interactionPenalty(int pi, int pj, int qi, int qj, Data *D, int labelP, int labelQ, OUTSIDE_MODE outsideMode) {
  //cout << "computing interaction penalty" << endl;
  Vec2d offsetP = D->offsets[labelP];
  Vec2d offsetQ = D->offsets[labelQ];
  Mat SLP = D->sources[labelP];
  Mat SLQ = D->sources[labelQ];
  Mat GXLP = D->gradientXSources[labelP];
  Mat GYLP = D->gradientYSources[labelP];
  Mat GXLQ = D->gradientXSources[labelQ];
  Mat GYLQ = D->gradientYSources[labelQ];
  double termP, termQ, termPx, termPy, termQx, termQy;
  if (!Collage::isInImage(pi, pj, offsetQ[0], offsetQ[1], SLQ) || !Collage::isInImage(qi, qj, offsetP[0], offsetP[1], SLP) ||
  	!Collage::isInImage(pi, pj, offsetP[0], offsetP[1], SLP) || !Collage::isInImage(qi, qj, offsetQ[0], offsetQ[1], SLQ)) {
  	return 0;
  }

  switch(outsideMode) {
  	case COLORS:
  //	cout << pi << ", " << pj << endl;
  //	cout << "for colors" << pi - offsetP[0] << ", " << pj - offsetP[1] <<endl;
  		termP = norm((Vec3d)SLP.at<Vec3b>(pi - offsetP[0], pj - offsetP[1])-(Vec3d)SLQ.at<Vec3b>(pi - offsetQ[0], pj - offsetQ[1])); 
  		termQ = norm((Vec3d)SLP.at<Vec3b>(qi - offsetP[0], qj - offsetP[1])-(Vec3d)SLQ.at<Vec3b>(qi - offsetQ[0], qj - offsetQ[1])); 
  		return termP + termQ;

  	case GRADIENTS:
  	//cout << "for gradients" << endl;
  		termPx = norm(GXLP.at<Vec3d>(pi - offsetP[0], pj - offsetP[1])-GXLQ.at<Vec3d>(pi - offsetQ[0], pj - offsetQ[1]));
  		termPy = norm(GYLP.at<Vec3d>(pi - offsetP[0], pj - offsetP[1])-GYLQ.at<Vec3d>(pi - offsetQ[0], pj - offsetQ[1]));
  		termQx = norm(GXLP.at<Vec3d>(qi - offsetP[0], qj - offsetP[1])-GXLQ.at<Vec3d>(qi - offsetQ[0], qj - offsetQ[1]));  	
  		termQy = norm(GYLP.at<Vec3d>(qi - offsetP[0], qj - offsetP[1])-GYLQ.at<Vec3d>(qi - offsetQ[0], qj - offsetQ[1]));
  		return sqrt(termPx * termPx + termPy * termPy)  + sqrt(termQx * termQx + termQy * termQy);
  	case COLORS_AND_GRADIENTS:
  		return interactionPenalty(pi, pj, qi, qj, D, labelP, labelQ, COLORS) +
  			interactionPenalty(pi, pj, qi, qj, D, labelP, labelQ, GRADIENTS);
  }	
}

/*	Crée le graphe des flots pour une image  en fonction des modes */
void computeGraph(Graph<double, double, double> &G, INSIDE_MODE insideMode, OUTSIDE_MODE outsideMode, const Mat &R0, Data *D, int alpha) {
	G.add_node(D->height * D->width); //les premiers noeuds seront les pixels, les suivants les voisinages entre deux pixels
	int middleNode = D->height * D->width;

	for (int i = 0; i < D->height; i++) {
		for (int j = 0; j < D->width; j++) {
			double currentImage = R0.at<uchar>(i, j);
			double capacityToPuits = dataPenalty(alpha, currentImage, D, i, j, insideMode);
			double capacityToSource = INT_MAX;
			if (i - D->offsets[alpha][0] >= 0 && j - D->offsets[alpha][1] >= 0) {
				capacityToSource = dataPenalty(alpha, alpha, D, i, j, insideMode);
			}

			G.add_tweights(D->width * i + j, capacityToSource, capacityToPuits);

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
