/*
 * expression.cpp
 *
 *  Created on: 14.6.2013
 *      Author: Jyrki Numminen
 *      A JNI interface for expression_recognizer.cpp for native OpenCV part in Android
 */

#include <opencv2/core/core.hpp>

#include <string>
#include <vector>
#include <android/log.h>

#include "expression.h"
#include "expressionrecognizer.h"

#define LOG_TAG "ExpressionJNI"
#define LOGD(...) ((void)__android_log_print(ANDROID_LOG_DEBUG, LOG_TAG, __VA_ARGS__))
#define LOGE(...) ((void)__android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__))

using namespace std;
using namespace cv;

#define catcher(x)	catch(cv::Exception& e) { LOGE(x); LOGE("cv::Exception: %s", e.what());\
					jclass je = jenv->FindClass("org/opencv/core/CvException");\
					if(!je)	je = jenv->FindClass("java/lang/Exception");\
					jenv->ThrowNew(je, e.what()); } catch (...) { LOGE(x);\
					LOGE("caught unknown exception");\
					jclass je = jenv->FindClass("java/lang/Exception");\
					jenv->ThrowNew(je, "Unknown exception in JNI code");}

inline void vector_Rect_to_Mat(vector<Rect>& v_rect, Mat& mat)
{
    mat = Mat(v_rect, true);
}

JNIEXPORT jlong JNICALL Java_com_siperia_peopleinphotos_expression_nativeCreateObject
(JNIEnv * jenv, jclass)
{
	jlong result = 0;
	try
	{
		result = (jlong)new expression_recognizer();
	}
	catcher("nativeCreateObject")

	return result;
}

JNIEXPORT void JNICALL Java_com_siperia_peopleinphotos_expression_nativeDestroyObject
(JNIEnv * jenv, jclass, jlong thiz)
{
	try
	{
		if(thiz != 0)
		{
			delete (expression_recognizer*)thiz;
		}
	}
	catcher("nativeDestroyObject")
}

JNIEXPORT void JNICALL Java_com_siperia_peopleinphotos_expression_nativeInitModels
(JNIEnv * jenv, jclass, jlong thiz, jint fisherfaces, jdouble fisherconf, jint eigenfaces, jdouble eigenconf, jboolean load ) {
	try {
		((expression_recognizer*)thiz)->initModels(fisherfaces, fisherconf, eigenfaces, eigenconf, load);
	}
	catcher("nativeInitModels")
}

JNIEXPORT void JNICALL Java_com_siperia_peopleinphotos_expression_nativeAddFisherface
(JNIEnv * jenv, jclass, jlong thiz, jlong jFace, jint jClassification ) {
	try {
		((expression_recognizer*)thiz)->addFisherface(*((Mat*)jFace), jClassification);
	}
	catcher("nativeAddFisherface")
}

JNIEXPORT void JNICALL Java_com_siperia_peopleinphotos_expression_nativeTrainFisherfaces
(JNIEnv * jenv, jclass, jlong thiz ) {

	try {
		((expression_recognizer*)thiz)->trainFisherfaces();
	}
	catcher("nativeTrainFisherfaces")
}

JNIEXPORT jint JNICALL Java_com_siperia_peopleinphotos_expression_nativePredictFisherface
(JNIEnv * jenv, jclass, jlong thiz, jlong jFace ) {

	int retval = 255;
	try {
		retval = ((expression_recognizer*)thiz)->predictFisherface(*((Mat*)jFace));
	}
	catcher("nativePredictFisherface")
	return retval;
}

JNIEXPORT void JNICALL Java_com_siperia_peopleinphotos_expression_nativeAddEigenface
(JNIEnv * jenv, jclass, jlong thiz, jlong jFace, jint jClassification ) {
	try {
		((expression_recognizer*)thiz)->addEigenface(*((Mat*)jFace), jClassification);
	}
	catcher("nativeAddEigenface")
}

JNIEXPORT void JNICALL Java_com_siperia_peopleinphotos_expression_nativeTrainEigenfaces
(JNIEnv * jenv, jclass, jlong thiz ) {

	try {
		((expression_recognizer*)thiz)->trainEigenfaces();
	}
	catcher("nativeTrainEigenfaces")
}

JNIEXPORT jint JNICALL Java_com_siperia_peopleinphotos_expression_nativePredictEigenface
(JNIEnv * jenv, jclass, jlong thiz, jlong jFace ) {

	int retval = 255;
	try {
		retval = ((expression_recognizer*)thiz)->predictEigenface(*((Mat*)jFace));
	}
	catcher("nativePredictEigenface")
	return retval;
}

JNIEXPORT void JNICALL Java_com_siperia_peopleinphotos_expression_nativeEdgeHistogram
(JNIEnv * jenv, jclass, jlong thiz, jlong jFace, jlong jHist) {
	try {
		((expression_recognizer*)thiz)->edgeHistogram(*((Mat*) jFace),*((Mat*) jHist));
	}
	catcher("nativeEdgeHistogram")
}

JNIEXPORT void JNICALL Java_com_siperia_peopleinphotos_expression_nativeSkinThreshold
(JNIEnv * jenv, jclass, jlong thiz, jlong jFrame, jlong clip, jboolean scan)
{
	//float rval = 0.0f;
	try {
		vector<Rect> RectFaces;
		((expression_recognizer*)thiz)->skinThreshold(*((Mat*)jFrame), RectFaces, scan);
		vector_Rect_to_Mat(RectFaces, *((Mat*)clip));
	}
	catcher("nativeSkinThreshold")
	//return rval;
}

JNIEXPORT void JNICALL Java_com_siperia_peopleinphotos_expression_nativeLocalMeanThreshold
(JNIEnv * jenv, jclass, jlong thiz, jlong jFaceMat, jint A, jint B, jint P, jfloat phase) {
	try {
		((expression_recognizer*)thiz)->localMeanThreshold(*((Mat*)jFaceMat), A, B, P, phase);
	}
	catcher("nativeLocalMeanThreshold")
}

JNIEXPORT void JNICALL Java_com_siperia_peopleinphotos_expression_nativeConcatHist
(JNIEnv * jenv, jclass, jlong thiz, jlong jMat1, jlong jMat2, jlong jHist) {
	try {
		((expression_recognizer*)thiz)->concatHist(*((Mat*)jMat1), *((Mat*)jMat2), *((Mat*)jHist));
	}
	catcher("nativeConcatHist")
}

JNIEXPORT void JNICALL Java_com_siperia_peopleinphotos_expression_nativeELBP
(JNIEnv * jenv, jclass, jlong thiz, jlong jFaceMat, jint A, jint B, jint P, jfloat phase) {
	try {
		((expression_recognizer*)thiz)->ELBP(*((Mat*)jFaceMat),A,B,P,phase);
	}
	catcher("nativeELBP")
}


JNIEXPORT void JNICALL Java_com_siperia_peopleinphotos_expression_nativeARLBP
(JNIEnv * jenv, jclass, jlong thiz, jlong jFaceMat, jlong jHistogram, jlong jsHist, jint vdivs, jint hdivs)
{
	try {
		((expression_recognizer*)thiz)->ARLBP(*((Mat*)jFaceMat), *((Mat*)jHistogram), *((Mat*)jsHist), vdivs, hdivs);
	}
	catcher("nativeARLBP")
}

JNIEXPORT void JNICALL Java_com_siperia_peopleinphotos_expression_nativeGaborLBPHistograms
(JNIEnv * jenv, jclass, jlong thiz, jlong jFaceMat, jlong jHistogram, jlong jLUT, jint N, jint step, jint ind)
{
	try {
		((expression_recognizer*)thiz)->gaborLBPHistograms(*((Mat*)jFaceMat), *((Mat*)jHistogram),
				*((Mat*)jLUT), N, step, ind);
	}
	catcher("GaborLBP_Histograms")
}

JNIEXPORT void JNICALL Java_com_siperia_peopleinphotos_expression_nativeSetFilterSize
(JNIEnv * jenv, jclass, jlong thiz, jint height, jint width)
{
	CV_Assert(height > 0);
	CV_Assert(width > 0);
	try {
		if ((height > 0) && (width > 0))
		{
			((expression_recognizer*)thiz)->setFilterSize(height, width);
		}
	}
	catcher("nativeSetFilterSize")
}

JNIEXPORT void JNICALL Java_com_siperia_peopleinphotos_expression_nativeGetEigenPictures
(JNIEnv * jenv, jclass, jlong thiz, jlong a, jlong b, jlong c, jlong d, jlong e)
{
	try {
		((expression_recognizer*)thiz)->getEigenPictures(*((Mat*)a),*((Mat*)b),*((Mat*)c),*((Mat*)d),*((Mat*)e));
	}
	catcher("GetEigenPictures")
}

