/*
 * expressionrecognizer.cpp
 *
 *  Created on: 14.6.2013
 *      Jyrki Numminen
 */

#include "expressionrecognizer.h"

#include <android/log.h>
#include <math.h>

#define LOG_TAG "Expression recognizer"
#define LOGD(...) ((void)__android_log_print(ANDROID_LOG_DEBUG, LOG_TAG, __VA_ARGS__))

#define genderFisherFile "/storage/sdcard0/Pictures/PiP_idents/genderFisherfaces.xml"
#define eigenfacesFile "/storage/sdcard0/Pictures/PiP_idents/eigenfaces.xml"

using namespace cv;

expression_recognizer::expression_recognizer() :
		height(1), width(1), cutoffPoint_8(0), cutoffPoint_10(0), currA(0), currB(0), currP(0), currPhase(0) {

	for (unsigned int i=0;i<8;i++) area[i]=0.0f;
	trained = false;

	// building of uniform LBP index lookup table.
	// A uniform pattern has exactly 2 binary transitions in it, i.e. 00111000, 11110111, 00000010 etc
	Mat LUT_8 = Mat(1, 256, CV_8U, 255);
	Mat LUT_10 = Mat(1, 1024, CV_16U, 1023);
	unsigned int count_8=0, count_10=0;
	for (unsigned int i=0;i<1024;i++) {
		unsigned int trans_8=0, trans_10=0;

		for (unsigned int bit=0;bit<9;bit++) { //compare bits 1-7 to bits 2-8 etc
			bool transition = ((i & powtable[bit]) > 0) != ((i & powtable[bit+1]) > 0);
			if ( transition && (bit < 7) ) trans_8++;
			if ( transition ) trans_10++;
		}

		if (i < 256) {
			if (((i& powtable[0])>0) != ((i&powtable[7])>0)) trans_8++; // cyclic
			if (trans_8 <= 2) {
				LUT_8.at<uchar>(0,i) = count_8;
				cutoffPoint_8 = ++count_8;
			}
		}

		if ( ((i& powtable[0])>0) != ((i&powtable[9])>0)) trans_10++;
		if (trans_10 <= 2) {
			LUT_10.at<unsigned short>(0,i) = count_10;
			cutoffPoint_10 = ++count_10;
		}
	}

	uniforms_8 = LUT_8.clone();
	uniforms_10 = LUT_10.clone(),
	LOGD("debug: Uniform patterns: %u(8b) and %u(10b)", cutoffPoint_8, cutoffPoint_10);

	generate_gabor_kernels();

}

expression_recognizer::~expression_recognizer() {
}

void expression_recognizer::initModels( int fisherfaces, double fisher_conf, int eigenfaces, double eigen_conf, bool load ) {
	fisherface_model.release();
	eigenface_model.release();

	fisherface_model = createFisherFaceRecognizer(fisherfaces, fisher_conf);
	eigenface_model = createEigenFaceRecognizer(eigenfaces, eigen_conf);

	if (load) {
		fisherface_model->load(genderFisherFile);
		eigenface_model->load(eigenfacesFile);
	}
}

void expression_recognizer::addFisherface(cv::Mat& face, int classification) {
	fisherSamples.push_back(face.clone());
	fisherClassifications.push_back(classification);
}

void expression_recognizer::trainFisherfaces() {
	fisherface_model->train(fisherSamples, fisherClassifications);
	fisherface_model->save(genderFisherFile);

	fisherSamples.clear();
	fisherClassifications.clear();
}

int expression_recognizer::predictFisherface( Mat& face ) const {
	return fisherface_model->predict( face );
}

void expression_recognizer::addEigenface(Mat& face, int classification) {
	eigenSamples.push_back(face.clone());
	eigenClassifications.push_back(classification);
}

void expression_recognizer::trainEigenfaces() {
	eigenface_model->train(eigenSamples, eigenClassifications);
	eigenface_model->save(eigenfacesFile);
	eigenSamples.clear();
	eigenClassifications.clear();
	//so we can just append new training data to start a new training process
}

int expression_recognizer::predictEigenface( Mat& face ) const {
	return eigenface_model->predict( face );
}

void expression_recognizer::getEigenPictures( Mat& a, Mat& b, Mat& c, Mat& d, Mat& e) {
	a = eigenface_model->getMat("mean");
	Mat W = eigenface_model->getMat("eigenvectors");
	b = W.col(0).clone();
	c = W.col(1).clone();
	d = W.col(2).clone();
	e = W.col(3).clone();
}

// returns area% of detected skin
void expression_recognizer::skinThreshold(Mat& frame, vector<Rect>& result, bool scan) {

	CV_Assert(frame.size().width > 0);
	CV_Assert(frame.size().height > 0);
	//CV_Assert(frame.channels() == 3);

	int width = frame.size().width;
	int height = frame.size().height;
	unsigned int count = 0;

	// this block is here only to demonstrate how this would be done using ready made OpenCV functions
	if (false) {
		planes.clear();
		split(frame, planes);
		Mat red = planes[0];
		Mat green = planes[1];
		Mat blue = planes[2];

		absdiff(red, green, mask);
		threshold(mask, mask, 15, 1, THRESH_BINARY); // |R-G|>15
		// saves 1 add this way

		threshold(red, help1, 95, 1, THRESH_BINARY); // r > 95
		add(help1,mask,mask);
		threshold(green, help1, 40, 1, THRESH_BINARY); // g > 40
		add(help1,mask,mask);
		threshold(blue, help1, 20, 1, THRESH_BINARY); // b > 20
		add(help1,mask,mask);

		max(green, blue, help2);
		subtract(red, help2, help1);
		threshold(help1, help1, 0, 1, THRESH_BINARY); // R > G, R > B
		add(help1, mask, mask);

		max(help2, blue, help2); // help2 is now max(r,g,b)
		min(red, green, help1);
		min(help1, blue, help1); // and red is min(r,g,b);
		subtract(help2, help1, help1); // max-min
		threshold(help1, help1, 15, 1, THRESH_BINARY);
		add(help1, mask, mask);

		// now the pixels which fullfill every condition will have value of 6
		// (6 binary additive steps to mask)
		threshold(mask, mask, 5, 255, THRESH_BINARY); // > 5
	} else {
		// This method should get slightly better performance due to caching
		mask = Mat::zeros(frame.size(), CV_8U);
		int step0 = frame.step[0], step1 = frame.step[1];
		int red=0,green=1,blue=2;
		int min=0, max=0;

		for (unsigned int y = 0; y < height; y++) {
			for (unsigned int x = 0; x < width; x++) {
				uchar* dp = frame.data+step0*y+step1*x;

				if (dp[red] > 95 && dp[green] > 40 && dp[blue] > 20)
				if (abs(dp[red]-dp[green]) > 15) {
					if (dp[red] < dp[green] && dp[red] < dp[blue]) { min = dp[red]; }
					else if (dp[green] < dp[red] && dp[green] < dp[blue]) { min = dp[green]; }
					else min = dp[blue];
					if (dp[red] > dp[green] && dp[red] > dp[blue]) {max = dp[red];}
					else if (dp[green] > dp[red] && dp[green] > dp[blue]) {max = dp[green];}
					else max = dp[blue];
					if (max-min > 15) {
						mask.at<uchar>(y,x) = 255;
						count++;
					}
				}
			}
		}
	}

	unsigned int minx=0,miny=0,maxx=width-1,maxy=height-1;
	if (scan) {
		static int morph_size = 6;
		static Mat element = getStructuringElement( 0, Size( 2*morph_size + 1, 2*morph_size+1 ), Point( morph_size, morph_size ) );
		//morphologyEx( mask, mask, MORPH_OPEN, element);
		erode(mask,mask,element);
		//erode(mask,mask,element);

		// To get a useable face sample the face must be fully in frame.
		// Practical minimum face size is 24x24 so search by grid
		bool found[] = {false,false,false,false};
		static int frame_margin = 12; //Distance to ignore near frame borders
		static int grid_size = 12; //How dense the search grid is, should be at least 1/2 of minimum useable face size
		static int padding = 12;  // how much the found area is padded from the sides. Without padding a square
								 // classifier can't see most faces, especially if only a part of the face is seen
								 // as skin colored.

		for (unsigned int x = frame_margin;x < width-frame_margin; x+=grid_size) {
			for (unsigned int y = frame_margin; y < height-frame_margin; y+=grid_size) {
				if (!found[0] && (mask.at<uchar>(y,x) > 0)) { minx = x-padding; found[0]=true; }
				if (!found[1] && (mask.at<uchar>(y,width-x-1) > 0)) { maxx = width-x+padding; found[1]=true; }

				if (found[0] && found[1]) goto findrows;
			}
		}
		findrows: for (unsigned int y = frame_margin;y < height-frame_margin; y+=grid_size) {
			for (unsigned int x = frame_margin; x < width-frame_margin; x+=grid_size) {
				if (!found[2] && (mask.at<uchar>(y,x) > 0)) { miny = y-padding; found[2]=true; }
				if (!found[3] && (mask.at<uchar>(height-y-1,x) > 0)) { maxy = height-y+padding;	found[3]=true; }

				if (found[2] && found[3]) goto donesearch;
			}
		}
	}

	donesearch:clip.clear();

	clip.push_back( Rect(MAX(minx,0),MAX(miny,0),MIN(maxx-minx, width-1),MIN(maxy-miny, height-1) ) );
	clip.push_back( Rect(count,width*height,0,0) );

	result = clip;
}

// First part of this implementation is adapted from the original:
// http://www.ee.oulu.fi/~topiolli/cpplibs/files/LBP.c
// Interpolation deemed less important as this is used in method where sides are clipped off
// A = horizontal axis, B = vertical axis, P = steps. A=1,B=1,P=8 generates the normal 3x3 LBP
void expression_recognizer::updateELBPkernel( int A, int B, int P, float phase ) {
	float step = (M_PI * 2) / P;

	for (unsigned int i = 0; i < P; i++) {
		float tmpX = A * cos(i * step + phase);
		float tmpY = B * sin(i * step + phase);
		ELBP_coords[i].x = (int)tmpX;
		ELBP_coords[i].y = (int)tmpY;
		offsets[i].x = tmpX - ELBP_coords[i].x;
		offsets[i].y = tmpY - ELBP_coords[i].y;
		if (offsets[i].x < 1.0e-10 && offsets[i].x > -1.0e-10) /* rounding error */
			offsets[i].x = 0;
		if (offsets[i].y < 1.0e-10 && offsets[i].y > -1.0e-10) /* rounding error */
			offsets[i].y = 0;

		if (tmpX < 0 && offsets[i].x != 0) {
			ELBP_coords[i].x -= 1;
			offsets[i].x += 1;
		}
		if (tmpY < 0 && offsets[i].y != 0) {
			ELBP_coords[i].y -= 1;
			offsets[i].y += 1;
		}
	}
	currA = A; currB = B; currP = P; currPhase = phase;
}

#define access(T, m, y, x) if (x < 0 || x >= m.size().width || y < 0 || y >= m.size().height) { LOGD("ERROR %i", __LINE__); }

void expression_recognizer::ELBP(cv::Mat& face, unsigned int A, unsigned int B, unsigned int P, float phase) {
	CV_Assert(face.size().area() > 0);
	CV_Assert( A != 0 && B != 0 && P != 0);
	Mat ELBP( face.size(), CV_16UC1 );

	if (A != currA || B != currB || P != currP || currPhase != phase) updateELBPkernel( A,B,P,phase );

	unsigned int fw = face.size().width;
	unsigned int fh = face.size().height;

	for (unsigned int y=0;y<fh;y++) {
		for (unsigned int x=0;x<fw;x++) {
			unsigned short ELBP_code = 0;
			for (unsigned int i=0; i<P; i++) {
				unsigned int relx = ELBP_coords[i].x+x;
				unsigned int rely = ELBP_coords[i].y+y;
				access(uchar, face, y, x);
				uchar center = face.at<uchar>(y,x);

				if (relx >= 0 && relx < fw) {
					if (rely >= 0 && rely < fh) {
						access(uchar, face, rely, relx);
						if (face.at<uchar>(rely,relx) >= center) ELBP_code += powtable[i];
					}
				}
			}
			access(unsigned short, ELBP, y, x);
			ELBP.at<unsigned short>(y,x) = ELBP_code;
		}
	}

	face = ELBP;
}

// prepare a photo with CLBP_M (Completed LBP-Magnitude)
void expression_recognizer::localMeanThreshold(Mat& face, int A, int B, int P, float phase) {
	CV_Assert( face.depth() == CV_8U );
	CV_Assert( face.size().width == 48 && face.size().height == 60 );
	CV_Assert( A > 0 && B > 0 && P > 0 );

	if (A != currA || B != currB || P != currP || currPhase != phase) updateELBPkernel( A,B,P,phase );

	unsigned int fw = face.size().width;
	unsigned int fh = face.size().height;
	unsigned int marginal = (currA >= currB) ? currA : currB;

	Mat local_mean = Mat(face.size(), CV_32F);
	float total_mean=0;

	copyMakeBorder(face,face,marginal,marginal,marginal,marginal,BORDER_REPLICATE); //replication difference = 0

	// local difference mean per pixel
	for (unsigned int y = marginal; y<fh+marginal; y++) {
		for (unsigned int x = marginal; x<fw+marginal; x++) {
			float sum=0;
			access(uchar, face,y, x);
			uchar center = face.at<uchar>(y,x);

			for (unsigned int i=0; i<currP; i++) {
				unsigned int relx = ELBP_coords[i].x+x;
				unsigned int rely = ELBP_coords[i].y+y;

				access( uchar, face, rely, relx )
				uchar cmp = face.at<uchar>(rely,relx);
				sum += (float)abs(center - cmp);
			}

			sum /= (float)currP;
			access( float, local_mean, y-marginal, x-marginal )
			local_mean.at<float>(y-marginal,x-marginal) = sum;
			total_mean += sum;
		}
	}
	total_mean /= local_mean.size().area();
	//LOGD("debug: total mean %f", total_mean);

	// ELBP
	face.convertTo(face, CV_16U);
	unsigned short ELBP_code = 0;
	copyMakeBorder(local_mean,local_mean,marginal,marginal,marginal,marginal,BORDER_REPLICATE);

	for (unsigned int y=marginal;y<fh+marginal;y++) {
		for (unsigned int x=marginal;x<fw+marginal;x++) {
			ELBP_code = 0;

			for (unsigned int i=0; i<currP; i++) {
				unsigned int relx = ELBP_coords[i].x+x;
				unsigned int rely = ELBP_coords[i].y+y;

				access( float, local_mean, rely, relx )
				if (local_mean.at<float>(rely,relx) >= total_mean) ELBP_code += powtable[i];
			}

			access( unsigned short, face, y-marginal, x-marginal )
			face.at<unsigned short>(y-marginal,x-marginal) = ELBP_code;
		}
	}

	face = face.colRange(0, fw).rowRange(0, fh);
}

void expression_recognizer::uniformalize10( Mat& mat ) {
	CV_Assert( mat.size().height > 0 && mat.size().width == 1 ); // a histogram
	//LOGD("debug: histsize: %u"+mat.size().height);

	for (unsigned int i=0;i<mat.size().height;i++) {
		uchar loc = mat.at<uchar>(i,0);
		mat.at<unsigned short>(i,0) = uniforms_10.at<unsigned short>(0,loc);
	}
	mat = mat.rowRange(0,cutoffPoint_10).clone();
}

void expression_recognizer::concatHist(Mat& mat1, Mat& mat2, Mat& hist) {

	CV_Assert( mat1.type() == CV_16U ); // Straight ELBP 10b
	CV_Assert( mat2.type() == CV_16U ); // Means with 10b ELBP

	int hskip = (int)((float)mat1.size().width / 6);
	int vskip = (int)((float)mat1.size().height / 6);

	Mat tempHist;
	Mat firstHist, secondHist;
	int histSize[] = {65536};
	float range[] = {0, 65536};
	const float* histRange[] = {range};
	int channels[]={0};

	for (unsigned int y=0;y<6;y++) {
		for (unsigned int x=0;x<6;x++) {
			// select sub-region as region of interest
			Mat roi=mat1.rowRange(vskip*y, vskip*(y+1)).colRange(hskip*x,hskip*(x+1));
			// calculate the histogram of that ROI
			cv::calcHist(&roi, 1, 0, Mat(), tempHist, 1, histSize, histRange, true, false );
			//LUT(tempHist, uniforms_16, tempHist); // Oh, right.. LUT works only for 8bit tables.. *sigh*

			uniformalize10( tempHist );

			firstHist.push_back(tempHist.clone());
		}
	}

	hskip = (int)((float)mat2.size().width / 3);
	vskip = (int)((float)mat2.size().height / 3);

	for (unsigned int y=0;y<3;y++) {
		for (unsigned int x=0;x<3;x++) {
			Mat roi=mat2.rowRange(vskip*y, vskip*(y+1)).colRange(hskip*x,hskip*(x+1));
			cv::calcHist(&roi, 1, 0, Mat(), tempHist, 1, histSize, histRange, true, false );

			uniformalize10( tempHist );

			secondHist.push_back(tempHist.clone());
		}
	}

	// normalize separately
	normalize( firstHist, firstHist, 0, 1, NORM_MINMAX );
	normalize( secondHist, secondHist, 0, 1, NORM_MINMAX );
	hist.push_back( firstHist );
	hist.push_back( secondHist );

}


// Setter function for ARLBP filter size
void expression_recognizer::setFilterSize(int h_, int w_) {
	CV_Assert(h_ > 0 && w_ > 0);
	height = h_;
	width = w_;
}

// This function calculates different kinds of LBP descriptors used in most of the recognition
// methods used in this thesis. "face" is the incoming face picture. "hist" returns the result
// from ARLBP filtering where the result is split into hdivs*vdivs parts. The total length of
// histogram is therefore 256*hdivs*vdivs. "sHist" is regular 3x3 LBP filtered image histogram
// cut into only uniform patterns. The length is therefore (uniform patterns)*hdivs*vdivs in order
// of uniform patterns. (index-wise from smallest to largest)
void expression_recognizer::ARLBP(Mat& face, Mat& hist, Mat& sHist, int hdivs, int vdivs) {
	CV_Assert( face.depth() == CV_8U);
	CV_Assert( face.channels() == 1);
	CV_Assert( face.size() == cv::Size(64, 64) );
	CV_Assert( hdivs > 0 );
	CV_Assert( vdivs > 0 );

	// if height and width are set to 1 the function will only calculate simple, full-sized LBP-histograms
	bool smallLBP = ((height!=1) || (width!=1));

	// temporary holder for LBP picture
	Mat tempLBP;
	Mat tempLBPsmall;
	tempLBP.create(face.size(), CV_8UC1);
	if (smallLBP) tempLBPsmall.create(face.size(), CV_8UC1);

	copyMakeBorder(face,face,height,height,width,width,BORDER_REPLICATE,Scalar::all(0));

	// 1st. calculate the ARLBP picture with sliding window. Height and width parameters
	// are set by "setFilterSize" defined in the header file.
	uchar pixel=0;
	for (unsigned int y=0;y<64;y++) {
		 for (unsigned int x=0;x<64;x++) {
			 Mat roi = face.rowRange(Range(y, y+height+height+1)).colRange(Range(x, x+width+width+1));

			 // opencv/core.hpp MIN
			 int bb = MIN(face.rows-y-1,height);
			 int rb = MIN(face.cols-x-1,width);
			 int tb = MIN(y, height);
			 int lb = MIN(x, width); //border sizes

			 uchar center = roi.at<uchar>(height,width); // = face.at<uchar>(y+h,x+w);

			 for (unsigned int dx=1;dx<=width;dx++) {
				 for (unsigned int dy=1;dy<=height;dy++) {
					 area[0] += roi.at<uchar>(height-dy,width-dx);
					 area[2] += roi.at<uchar>(height-dy,width+dx);
					 area[4] += roi.at<uchar>(height+dy,width+dx);
					 area[6] += roi.at<uchar>(height+dy,width-dx);
					 if (dx == 1) { // any dx= 1 < constant < width
						 area[1] += roi.at<uchar>(height-dy,width);
						 area[5] += roi.at<uchar>(height+dy,width);
					 }
				 }
				 area[3] += roi.at<uchar>(height,width+dx);
				 area[7] += roi.at<uchar>(height,width-dx);
			 }
			 area[0] = cvRound((double)area[0] / MAX(1,(tb*lb)));
			 area[1] = cvRound((double)area[1] / MAX(1,tb));
			 area[2] = cvRound((double)area[2] / MAX(1,(rb*tb)));
			 area[3] = cvRound((double)area[3] / MAX(1,rb));
			 area[4] = cvRound((double)area[4] / MAX(1,(rb*bb)));
			 area[5] = cvRound((double)area[5] / MAX(1,bb));
			 area[6] = cvRound((double)area[6] / MAX(1,(lb*bb)));
			 area[7] = cvRound((double)area[7] / MAX(1,lb));

			 pixel=0;
			 for (unsigned int i=0;i<8;i++) {
				if (center <= area[i]) pixel += powtable[i];
				area[i] = 0;
			 }
			 tempLBP.at<uchar>(y,x) = pixel;

			 // calculate 3x3 sized LBP
			 if (smallLBP) {
				 pixel=0;
				 if (center <= roi.at<uchar>(height-1,width-1))	pixel += powtable[0];
				 if (center <= roi.at<uchar>(height-1,width))	pixel += powtable[1];
				 if (center <= roi.at<uchar>(height-1,width+1))	pixel += powtable[2];
				 if (center <= roi.at<uchar>(height,width+1))	pixel += powtable[3];
				 if (center <= roi.at<uchar>(height+1,width+1))	pixel += powtable[4];
				 if (center <= roi.at<uchar>(height+1,width))	pixel += powtable[5];
				 if (center <= roi.at<uchar>(height+1,width-1))	pixel += powtable[6];
				 if (center <= roi.at<uchar>(height,width-1))	pixel += powtable[7];
				 tempLBPsmall.at<uchar>(y,x) = pixel;
			 }
		 } // end x
	} // end y
	// for debugging
	//if (smallLBP) face = tempLBP.clone();
	face = face.colRange(width, 64+width).rowRange(height, 64+height);

	// Then concatenated histogram from the LBP pic.. 64x64 picture is split into 16 16x16 pictures.
	// Histograms are added in top->bottom, left->right order, as usual.

	//convert the small LBP to uniform patterns, every other value is set to 255 for easy removal
	if (smallLBP) LUT(tempLBPsmall, uniforms_8, tempLBPsmall);

	int histSize[] = {256};
	float range[] = {0, 256};
	const float* histRange[] = {range};
	int channels[]={0};
	Mat tempHist, histSmall;
	int hskip = (int)(tempLBP.size().width / hdivs);
	int vskip = (int)(tempLBP.size().height / hdivs);

	for (unsigned int y=0;y<vdivs;y++) {
		for (unsigned int x=0;x<hdivs;x++) {
			// select sub-region as region of interest
			Mat roi=tempLBP.rowRange(vskip*y, vskip*(y+1)).colRange(hskip*x,hskip*(x+1));
			// calculate the histogram of that ROI
			cv::calcHist(&roi, 1, 0, Mat(), tempHist, 1, histSize, histRange, true, false );
			// append to the histogram to be returned.
			hist.push_back(tempHist.clone());

			//and the same for 3x3 LBP
			if (smallLBP) {
				roi = tempLBPsmall.rowRange(vskip*y,vskip*(y+1)).colRange(hskip*x,hskip*(x+1));
				cv::calcHist(&roi, 1, 0, Mat(), histSmall, 1, histSize, histRange, true, false);
				histSmall = histSmall.rowRange(0,cutoffPoint_8); // clip off the non uniforms
				sHist.push_back(histSmall.clone());
			}
		}
	}

	normalize( hist, hist, 0, 1, NORM_MINMAX, -1, noArray());
	if (smallLBP) normalize( sHist, sHist, 0, 1, NORM_MINMAX, -1, noArray());
	// ..and leave without a fuss.
}

void expression_recognizer::generate_gabor_kernels(void)
{
	double f = sqrt(2.0);
	double sigma = M_PI*4;	 // M_PI*2*2
	double kmax = M_PI; 	 // M_PI*2/2
	register int x,y,u,v;

	const unsigned int mask_size_y = 64;
	const unsigned int mask_size_x = 64;

	Mat gabor_cos = Mat(mask_size_y,mask_size_x,CV_64F);
	Mat gabor_sin = Mat(mask_size_y,mask_size_x,CV_64F);
	Mat kernel;
	vector<Mat> complex;

	int offset_x = mask_size_x / 2;
	int offset_y = mask_size_y / 2;

	int m = getOptimalDFTSize( mask_size_y );
	int n = getOptimalDFTSize( mask_size_x );
    copyMakeBorder(gabor_cos, gabor_cos, 0, m - mask_size_y, 0, n - mask_size_x, BORDER_CONSTANT, Scalar::all(0));
    copyMakeBorder(gabor_sin, gabor_sin, 0, m - mask_size_y, 0, n - mask_size_x, BORDER_CONSTANT, Scalar::all(0));

	for (v=0;v<scale;v++)
    for (u=0;u<orientation;u++) {
		double kv=kmax/pow(f,v);
		double phiu=u*M_PI/8.0;
		double kv_mag=kv*kv;

		for (x = 0; x < mask_size_x; x++)
			for (y = 0; y< mask_size_y; y++) {
				int i=x-offset_x;
				int j=y-offset_y;
				double mag=(double)(i*i+j*j);
				gabor_cos.at<double>(y,x) = kv_mag/sigma*exp(-0.5*kv_mag*mag/sigma)*
							(cos(kv*(i*cos(phiu)+j*sin(phiu)))-exp(-1.0*sigma/2.0));
 				gabor_sin.at<double>(y,x) = kv_mag/sigma*exp(-0.5*kv_mag*mag/sigma)*
							(sin(kv*(i*cos(phiu)+j*sin(phiu))));//-exp(-1.0*sig/2.0)
 			}

		complex.clear();
		complex.push_back(gabor_cos);
		complex.push_back(gabor_sin);

		merge(complex, kernel);
		dft(kernel,kernel, DFT_COMPLEX_OUTPUT+ DFT_SCALE,0);

		gaborKernels.push_back(kernel.clone());
	}
	LOGD("debug: Kernels done");
}

// Filteres a face picture with 40 Gabor-filters, arranging the results into 3D-array-like
// structure and runs 3D-LBP operation on them. Outputs an idetification histogram after
// segmentation and concatenation in left-right, top-down order.
// If step is 0 or 1 returns a debug picture in "face". ind = index per (freq*orientation)+rot
void expression_recognizer::gaborLBPHistograms(Mat& face, Mat& hist, Mat& lut, int N, int step, int ind) {
	CV_Assert( face.depth() == CV_8U );
	CV_Assert( face.channels() == 1 );

	CV_Assert( lut.depth() == CV_8U );
	CV_Assert( lut.channels() == 1 );
	CV_Assert( lut.total() == 256 );
	CV_Assert( lut.isContinuous() );

	Mat tmppic = Mat( Size(64, 64), CV_64FC2);
	Mat holder = Mat( Size(64, 64), CV_64FC1);

	vector<Mat> planes;
	resize(face, face, Size(64,64));
	//face = face.colRange(8,80);//.clone();
	vector<Mat> doubleface;
	face.convertTo(face, CV_64FC2);

	int m = getOptimalDFTSize( face.size().height );
	int n = getOptimalDFTSize( face.size().width );
	copyMakeBorder(face, face, 0, m - face.size().height, 0, n - face.size().width, BORDER_CONSTANT, Scalar::all(0));

	doubleface.clear();
	doubleface.push_back(face);
	doubleface.push_back(face);
	merge( doubleface, face );

	dft(face, face, DFT_COMPLEX_OUTPUT + DFT_SCALE, 0);

	vector<Mat> gaborCube(scale*orientation);
	vector<Mat> binaryGaborVolume;
	for (unsigned int freq=0;freq<scale;freq++) {
		for (unsigned int rot=0;rot<orientation;rot++) {
			unsigned int index = (freq*orientation)+rot;

			Mat tmp = gaborKernels[index];
			mulSpectrums(face, tmp, tmppic, 0, false);
			idft(tmppic, tmppic, DFT_SCALE, 0);

			planes.clear();
			split(tmppic, planes);
			Mat p0=planes[0];
			Mat p1=planes[1];
			magnitude(p0,p1,holder);
			//holder = holder.colRange(0, 64).rowRange(0,64);
			// From real and imaginary parts we can get the magnitude for identification
			// add 1px borders for later, store in gabor-cube

			copyMakeBorder(holder, holder,1, 1, 1, 1, BORDER_CONSTANT, Scalar::all(0));
			gaborCube[index] = holder.clone();
		}
	}

	if (step == 0) face = gaborCube[ind];

	vector<Mat> LBP;
	Mat lbp = Mat(64,64,CV_8U);

	for (unsigned int freq=0;freq<scale;freq++) {
		for (unsigned int rot=0;rot<orientation;rot++) {

			unsigned int index = rot+(freq*orientation);
			Mat thiz = gaborCube[index];
			uchar pix = 0;

			for (unsigned int y=1;y<thiz.size().height-1;y++) {
				for (unsigned int x=1;x<thiz.size().width-1;x++) {
					pix = 0;
					double center = thiz.at<double>(y,x);

					// indices 1,3,5 and 7 are normal closest neighbor LBP
					if (thiz.at<double>(y-1,x) >= center ) pix += powtable[1];
					if (thiz.at<double>(y,x+1) >= center ) pix += powtable[3];
					if (thiz.at<double>(y+1,x) >= center ) pix += powtable[5];
					if (thiz.at<double>(y,x-1) >= center ) pix += powtable[7];

					// orientation neighbors are indices 2 and 6
					if (rot > 0) {
						Mat back = gaborCube[index-1];
						if ( back.at<double>(y,x) >= center ) pix += powtable[2];
					}
					if (rot < orientation-1) {
						Mat front = gaborCube[index+1];
						if ( front.at<double>(y,x) >= center ) pix += powtable[6];
					}
					//scale neighbors, indices 0,4
					if (freq > 0 ) {
						Mat back = gaborCube[index-orientation];
						if ( back.at<double>(y,x) >= center) pix += powtable[0];
					}

					if (freq < scale-1) {
						Mat front = gaborCube[index+orientation];
						if ( front.at<double>(y,x) >= center) pix += powtable[4];
					}

					lbp.at<uchar>(y-1,x-1) = pix;
				}
			}

			// 59 uniform patterns
			if (N>0) LUT(lbp, lut, lbp);

			LBP.push_back(lbp.clone());
		}
	}

	if (step == 1) face = LBP[ind];

	int histSize[] = {256};
	float range[] = {0, 256};
	const float* histRange[] = {range};
	int channels[]={0};

	static double areaWeights[] =		{  1,1,1,1,1,1,1,1,
								   	   	   1,1,1,1,1,1,1,1,
										   1,4,4,3,3,4,4,1,
										   1,4,4,3,3,4,4,1,
										   0,1,1,1,1,1,1,0,
										   0,1,2,2,2,2,1,0,
										   0,1,2,2,2,2,1,0,
										   0,0,1,1,1,1,0,0 };


	static unsigned int xstep=8, ystep=8;
	static unsigned int xsize=8, ysize=8;

	for (unsigned int y = 0;y<ystep;y++) {
		for (unsigned int x = 0;x<xstep;x++) {
			Mat accuhist = Mat::zeros(256,1,CV_32F);
			unsigned int weight = areaWeights[x+(y*xsize)];

			if (weight != 0) {
				for (unsigned int i=0;i<scale*orientation;i++) {
					Mat tempHist = Mat::zeros(256,1,CV_32F);
					lbp = LBP[i];

					Mat roi = lbp.rowRange(y*ysize, (y+1)*ysize).colRange(x*xsize,(x+1)*xsize);
					calcHist(&roi, 1, 0, Mat(), tempHist, 1, histSize, histRange, true, false );
					scaleAdd(tempHist, 1, accuhist, accuhist);
				}

				if (N>0) accuhist = accuhist.rowRange(0, N);
				// cut from 256 values per 8x8 area to 8 values per area
				//dump( accuhist );
				 hist.push_back(accuhist.clone());
				//cuts the ID vector length even more
			}
		}
	}

	normalize( hist, hist, 0, 1, NORM_MINMAX, -1, noArray());
}

void expression_recognizer::dump(Mat& mat) {
	string s("");
	Mat local;
	mat.convertTo(local, CV_64F);

	for (unsigned int y=0;y<mat.size().height;y++) {
		for (unsigned int x=0;x<mat.size().width;x++) {
			s.append(tostr( local.at<double>(y,x) ));
			s.append(" ");
		}
	}
	LOGD("debug dump: %s", s.c_str());
}

void expression_recognizer::edgeHistogram( Mat& face, Mat& hist ) {
	CV_Assert( face.depth() == CV_8U );
	CV_Assert( face.size() == Size(64, 64));
	CV_Assert( face.channels() == 1 );

	static unsigned int M = 10, N = 8;

	// calculate edge images
	Mat h_edges = Mat(face.size(), CV_16S);
	Mat v_edges = Mat(face.size(), CV_16S);
	Mat h_edges_pow = Mat( face.size(), CV_32F );
	Mat v_edges_pow = Mat( face.size(), CV_32F );

	Mat vertical = Mat(1,3,CV_16S);
	vertical.at<signed short>(0,0) = -1;
	vertical.at<signed short>(0,1) = 0;
	vertical.at<signed short>(0,2) = 1;
	Mat horizontal = vertical.t();

	filter2D( face, v_edges, CV_16S, vertical, Point(-1,-1), 0, BORDER_DEFAULT );
	filter2D( face, h_edges, CV_16S, horizontal, Point(-1,-1), 0, BORDER_DEFAULT );

	Mat magnitude = Mat( face.size(), CV_32F);
	Mat direction = Mat::zeros( face.size(), CV_8U );

	double angle = 0.0f;
	for (unsigned int y=0;y<face.size().height;y++) {
		for (unsigned int x=0;x<face.size().width;x++) {

			if (h_edges.at<signed short>(y,x) != 0) {
				if (v_edges.at<signed short>(y,x) != 0) {
					angle = atan2( v_edges.at<signed short>(y,x), h_edges.at<signed short>(y,x));
				} else angle = 0.0f;
				if (angle < 0.0f) angle += (2*M_PI);

				for (unsigned int n = N; n > 1; n--) {
					if (angle < (2*M_PI*n/N)) direction.at<uchar>(y,x) = n-1;
				}
			} else {
				direction.at<uchar>(y,x) = 0;
			}

			//if (x == 32) LOGD("angle: %u, %f", direction.at<uchar>(y,x), angle);
		}
	}

	v_edges.convertTo(v_edges, CV_32F);
	h_edges.convertTo(h_edges, CV_32F);
	pow( v_edges, 2, v_edges_pow );
	pow( h_edges, 2, h_edges_pow );
	add( h_edges_pow, v_edges_pow, magnitude );
	sqrt(magnitude, magnitude);

	normalize( magnitude, magnitude, 0, M+1-0.001, NORM_MINMAX, CV_8U);
	//normalize( direction, direction, 0, N-0.001, NORM_MINMAX, CV_8U); // so everything rounds down to 0-(N-1)

	/*LOGD("dir mag");
	dump( direction );
	dump( magnitude );*/

	unsigned int bin = 0;
	for (unsigned int y=0;y<face.size().height;y++) {
		for (unsigned int x=0;x<face.size().width;x++) {

			unsigned int m = magnitude.at<uchar>(y,x);
			unsigned int n = direction.at<uchar>(y,x);

			if (m == 0) bin = 0; else bin = ((m-1)*N)+n+1;

			face.at<uchar>(y,x) = bin; //histogram bins 0-81
			//if (x == 32) LOGD("debug: m:%u, n:%u = bin = %u", m, n, bin);
		}
	}

	int histSize[] = {256};
	float range[] = {0, 256};
	const float* histRange[] = {range};
	int channels[]={0};

	// split into 8x8 subimages
	static unsigned int ystep = 8, xstep = 8, xsize = 8, ysize = 8;
	for (unsigned int y = 0;y<ystep;y++) {
		for (unsigned int x = 0;x<xstep;x++) {
			// take histograms
			Mat tempHist = Mat(256,1,CV_8U);

			Mat roi = face.rowRange(y*ysize, (y+1)*ysize).colRange(x*xsize,(x+1)*xsize);
			calcHist(&roi, 1, 0, Mat(), tempHist, 1, histSize, histRange, true, false );

			tempHist = tempHist.rowRange(0, 82);
			hist.push_back(tempHist);
		}
	}

	normalize( hist, hist, 0, 1, NORM_MINMAX);
}
