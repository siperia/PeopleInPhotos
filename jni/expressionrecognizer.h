/*
 * expressionrecognizer.h
 *
 *  Created on: 14.6.2013
 *      Author: Jyrki Numminen
 */

#pragma once

#ifndef EXPRESSIONRECOGNIZER_H_
#define EXPRESSIONRECOGNIZER_H_

#include <opencv2/core/core.hpp>

#include <opencv2/core/types_c.h>
#include <opencv2/core/mat.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/ml/ml.hpp>
#include "opencv2/contrib/contrib.hpp"
#include <sstream>

using namespace cv;

template<typename coordtype>
struct coord{ coordtype x,y; };

static unsigned int powtable[] = {1,2,4,8,16,32,64,128,256,512,1024,2048,4096,8192, 16384, 32768};

class expression_recognizer {
private:
	int height;
	int width;
	unsigned int cutoffPoint_8;
	unsigned int cutoffPoint_10;
	unsigned int currA, currB, currP, currPhase; // current ELBP parameters
	coord<int> ELBP_coords[16];
	coord<int> offsets[16];

	unsigned int area[8];
	Mat uniforms_8, uniforms_10;

	Ptr<FaceRecognizer> fisherface_model;
	Ptr<FaceRecognizer> eigenface_model;

	vector<Mat> fisherSamples;
	vector<int> fisherClassifications;

	vector<Mat> eigenSamples;
	vector<int> eigenClassifications;

	CvSVM SVM;
	Mat training_data;
	Mat labels;
	bool trained;

	vector<Mat> planes;
	Mat mask;
	Mat help1;
	Mat help2;
	vector<Rect> clip;

	vector<Mat> gaborKernels;
	const static unsigned int scale = 5;
	const static unsigned int orientation = 8;
	const static unsigned int mask_size = 88;

	template <typename T> string tostr(const T& t) { std::ostringstream os; os<<t; return os.str(); };
	void dump(Mat& mat);

public:
	expression_recognizer();
	virtual ~expression_recognizer();

	virtual void setFilterSize(int h_, int w_);
	virtual void ARLBP(Mat& face_, Mat& hist_, Mat& sHist_, int vdivs , int hdivs );
	double interpolate_at_ptr(int* upperLeft, int i, int columns);
	void updateELBPkernel( int A, int B, int P, float phase );
	virtual void ELBP(Mat& face, unsigned int A, unsigned int B, unsigned int P, float phase);
	virtual void localMeanThreshold(Mat& matpic, int A, int B, int P, float phase);
	void uniformalize10( Mat& mat );
	virtual void skinThreshold( Mat& frame_,std::vector<cv::Rect>& result, bool scan );
	virtual void concatHist(Mat& jMat1, Mat& jMat2, Mat& jHist);

	virtual void initModels(int fisherfaces,double fisherconf,int eigenfaces,double eigenconf, bool load);

	virtual void addFisherface(Mat& face_, int classification_);
	virtual void trainFisherfaces();
	virtual int predictFisherface(Mat& face_) const;

	virtual void addEigenface(Mat& face, int classification);
	virtual void trainEigenfaces();
	virtual int predictEigenface( Mat& face ) const;

	virtual void edgeHistogram( Mat& face, Mat& hist );

	void generate_gabor_kernels();
	virtual void gaborLBPHistograms(Mat& face, Mat& hist, Mat& lut,int,int,int);

	virtual void getEigenPictures( Mat& a, Mat& b, Mat& c, Mat& d, Mat& e);
};

#endif /* EXPRESSIONRECOGNIZER_H_ */
