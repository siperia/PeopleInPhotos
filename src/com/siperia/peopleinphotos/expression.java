package com.siperia.peopleinphotos;

import org.opencv.core.Mat;
import org.opencv.core.MatOfRect;

public class expression
{
	public expression() {
   		mNativeObj = nativeCreateObject();
    }
	
	public void initModels( int fisherfaces, double fisherconf, int eigenfaces, double eigenconf) {
		nativeInitModels( mNativeObj, fisherfaces, fisherconf, eigenfaces, eigenconf);
	}
	
	public void addFisherface( Mat face, int classification ) {
		nativeAddFisherface( mNativeObj, face.getNativeObjAddr(), classification );
	}
	
	public void trainFisherfaces() {
		nativeTrainFisherfaces( mNativeObj );
	}
	
	public int predictFisherface( Mat face ) {
		return nativePredictFisherface( mNativeObj, face.getNativeObjAddr());
	}
	
	public void addEigenface( Mat face, int classification ) {
		nativeAddEigenface( mNativeObj, face.getNativeObjAddr(), classification );
	}
	
	public void trainEigenfaces() {
		nativeTrainEigenfaces( mNativeObj );
	}
	
	public int predictEigenface( Mat face ) {
		return nativePredictEigenface( mNativeObj, face.getNativeObjAddr());
	}
	
	public void ELBP( Mat face, int A, int B, int P, float phase) {
		nativeELBP( mNativeObj, face.getNativeObjAddr(), A, B, P, phase);
	}

    public void setFilterSize(int h, int w) {
        nativeSetFilterSize(mNativeObj, h, w);
    }

    public void ARLBP(Mat imageGray, Mat histogram, Mat sHistogram, int hdivs, int vdivs) {        
        nativeARLBP(mNativeObj, imageGray.getNativeObjAddr(), histogram.getNativeObjAddr(), sHistogram.getNativeObjAddr(), hdivs, vdivs);
    }
    
    public void skinThreshold(Mat mRgba, MatOfRect clip, boolean scan) {
    	nativeSkinThreshold( mNativeObj, mRgba.getNativeObjAddr(), clip.getNativeObjAddr(), scan);
    }
    
    public void localMeanThreshold(Mat matpic, int vdivs, int hdivs) {
    	nativeLocalMeanThreshold(mNativeObj, matpic.getNativeObjAddr(), vdivs, hdivs);
    }
    
    public void concatHist( Mat facepic, Mat means, Mat age_hist) {
    	nativeConcatHist(mNativeObj, facepic.getNativeObjAddr(), means.getNativeObjAddr(), age_hist.getNativeObjAddr());
    }
    
    public void GaborLBP_Histograms( Mat pic, Mat hist, Mat LUT, int N, int step, int ind ) {
    	nativeGaborLBPHistograms(mNativeObj, pic.getNativeObjAddr(), hist.getNativeObjAddr(),
    			LUT.getNativeObjAddr(), N, step, ind);
    }
    
    public void getEigenPictures( Mat a, Mat b, Mat c, Mat d, Mat e) {
    	nativeGetEigenPictures( mNativeObj, a.getNativeObjAddr(), b.getNativeObjAddr(),
    			c.getNativeObjAddr(), d.getNativeObjAddr(), e.getNativeObjAddr() );
    }

    public void release() {
        nativeDestroyObject(mNativeObj);
        mNativeObj = 0;
    }

    private long mNativeObj = 0;

    private static native long nativeCreateObject();
    private static native void nativeDestroyObject(long thiz);
    
    private static native void nativeInitModels( long thiz, int fisherfaces, double fisherconf,
    														int eigenfaces, double eigenconf);
    
    private static native void nativeAddFisherface( long thiz, long face, int classification );
    private static native void nativeTrainFisherfaces( long thiz );
    private static native int  nativePredictFisherface( long thiz, long face );
    
    private static native void nativeAddEigenface( long thiz, long face, int classification );
    private static native void nativeTrainEigenfaces( long thiz );
    private static native int  nativePredictEigenface( long thiz, long face );
    
    private static native void nativeELBP( long thiz, long face_, int A, int B, int P, float phase);
    private static native void nativeSetFilterSize(long thiz, int h, int w);
    private static native void nativeARLBP(long thiz, long inputImage, long histogram, long sHistogram, int vdivs, int hdivs);
    private static native void nativeSkinThreshold( long thiz, long mRgba, long clip, boolean scan);
    private static native void nativeLocalMeanThreshold( long thiz, long face, int vdivs, int hdivs);
    private static native void nativeConcatHist( long thiz, long facepic, long means, long age_hist);
    private static native void nativeGaborLBPHistograms( long thiz, long pic, long hist,long LUT, int N,int step, int ind);
    
    private static native void nativeGetEigenPictures( long thiz, long a, long b, long c, long d, long e);
    
}

