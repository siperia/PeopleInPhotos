package com.siperia.peopleinphotos;

import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import org.opencv.android.Utils;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfDouble;
import org.opencv.imgproc.Imgproc;

import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.util.Log;

public class Identity {
	
	private static final String TAG = "Identity class";
	private String username;
	private int ID;
	int gender;
	
	int[] features = null;
	private ArrayList<sample> faces = new ArrayList<sample>();
	private Mat classMeanGabor = null;
	public ArrayList<sample> getSampleList() { return faces; }
	
		
	Identity(String name, int IDno) {
		username = name;
		ID = IDno;
	}
	
	public String getName() { return username; }
	public int getID() { return ID; };
	
	public boolean equals(Object other) {
        if (other.equals(username)) return true;
        return false;
    }
	
	public void addSample(String absname) {
		faces.add(new sample( absname ));
	}
	public boolean contains( String absname ) {
		for (sample s: faces) {
			if (s.getFileName().equals( absname )) return true;
		}
		return false;
	}
	
	public List<double[]> getGaborVectors() {
		List<double[]> vtab = new ArrayList<double[]>();
		
		for (sample s: faces) {
			double[] list = new MatOfDouble(s.getGaborMat()).toArray();
			vtab.add( list );
		}
		
		return vtab;
	}
	
	public void setFeaturesAndReduce(int[] fin) {
		double[] filtered = new double[fin.length];
		features = new int[fin.length];
		features = fin;
		
		for (sample s: faces) {
			double[] allGabor = new MatOfDouble(s.getGaborMat()).toArray();
			for (int u = 0;u < fin.length; u++) {
				filtered[u] = allGabor[fin[u]];
			}
			s.setFiltered(filtered);
		}
	}
	
	public void calculateMeanGabor() {
		// FFS openCV !
		double[] meanface = null;
		for (sample s: faces) {
			double[] tmp = s.getFilteredGabor().toArray();
			if (meanface == null) meanface = new double[tmp.length];
			
			for (int k=0;k<tmp.length;k++) meanface[k]+=tmp[k];
		}
		for (int k=0;k<meanface.length;k++) meanface[k] /= faces.size();
		classMeanGabor = new MatOfDouble(meanface);
		
		//Log.d(TAG, classMeanGabor + "cMG:"+classMeanGabor.dump());
	}
	public Mat getMeanVector() {
		return classMeanGabor;
	}
	
	public int[] getFeatures() { return features; }
	public int getNumberOfSamples() { return faces.size(); }
	
	public void updateToLDA(Mat LDA) {
		for (sample s: faces) {
			MatOfDouble facevector = s.getFilteredGabor();
			MatOfDouble result = new MatOfDouble();
			result.create(facevector.size(), CvType.CV_32F);
			Core.gemm( LDA, facevector, 1, new Mat(), 0, result, Core.GEMM_1_T);
			s.setLDA(result);
		}
	}
	
	public void setGender(int gin_) { gender = gin_; }
	public int getGender() { return gender; }
	
	public class sample {
		String fileName;
		Mat rawPicture = new Mat();
		Mat eigenface = new Mat();
		Mat gaborMat = new Mat();
		MatOfDouble gaborFiltered = null;
		float[] binarizedGabor = null;
		MatOfDouble LDAmat = null;
		
		sample(String filename) {
			fileName = filename;
			Log.d(TAG, "added sample "+filename+", size:"+rawPicture.size());
		}
		
		public void reloadPicture() {
			Log.d(TAG, "reloading "+fileName);
			if (fileName != null) {
				File photoFile = new File( fileName );
				if (photoFile.exists()) {					
					Bitmap bm = null;
					try {
						FileInputStream fis = new FileInputStream(photoFile);
						bm = BitmapFactory.decodeFile(photoFile.getAbsolutePath());
						fis.close();
					} catch (IOException e) {
			            e.printStackTrace();
			            Log.e(TAG, "IOException: " + e);
			        }
					Utils.bitmapToMat(bm, rawPicture);
					Imgproc.cvtColor(rawPicture, rawPicture, Imgproc.COLOR_RGB2GRAY);
					Imgproc.equalizeHist(rawPicture, rawPicture);
				}	
			}
		}
		
		public String getFileName() { return fileName; }
		
		public boolean equals(Object other) {
	        if (other.equals(fileName)) return true;
	        return false;
	    }
				
		public Mat getGaborMat() { return gaborMat; }
		
		public MatOfDouble getFilteredGabor() { return gaborFiltered; }
		public void setFiltered( double[] filtered ) {
			gaborFiltered = new MatOfDouble(filtered);
		}		
		
		public Mat getEigenface() { return eigenface; }
		public void setEigenface(Mat ef) { eigenface = ef; }
		
		public void setBinarizedGaborVector( float[] d ) {
			binarizedGabor = new float[d.length];
			binarizedGabor = d.clone();
		}
		public float[] getBinarizedGaborVector() { return binarizedGabor; }
		
		public Mat getRawPicture() { return rawPicture; }
		public void setRawPicture( Mat raw ) { rawPicture = raw; }
		public void releasePicture() { rawPicture = null; }
		
		
		public void setGaborMat(Mat mat) {
			gaborMat = mat;
		}
		public void setGaborVector(double[] gabor) {
			gaborMat = new MatOfDouble(gabor);
		}
		
		public void setLDA( MatOfDouble LDA ) {
			LDAmat = LDA;
		}
		public MatOfDouble getLDA () {
			return LDAmat;
		}
	}
}
