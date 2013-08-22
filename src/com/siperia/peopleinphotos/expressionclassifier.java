package com.siperia.peopleinphotos;

import java.io.File;
import java.io.FileInputStream;
import java.io.FilenameFilter;
import java.io.IOException;
import java.util.Scanner;

import org.opencv.android.Utils;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfInt;
import org.opencv.core.MatOfRect;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.core.TermCriteria;
import org.opencv.imgproc.Imgproc;
import org.opencv.ml.CvSVM;
import org.opencv.ml.CvSVMParams;
import org.opencv.objdetect.Objdetect;

import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.os.Environment;
import android.util.Log;
import android.widget.Toast;

import com.siperia.peopleinphotos.Identity;
import com.siperia.peopleinphotos.expression;
import com.siperia.peopleinphotos.Identity.sample;

public class expressionclassifier {
	private static final String		TAG="ExpressionClassifier";
	private static final File 	   	sdDir = Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_PICTURES);
	private static final File	   	identRootDir = new File(sdDir, "PiP_idents");
	private static final File	   	otherPip = new File(sdDir, "PiP");
	private static final File    	facesDir = new File(otherPip, "FACES");
	private static final String    	expressionSVMname = identRootDir + "/emotionSVM.xml";
	private static final String	   	genderSVMname = identRootDir + "/genderSVM.xml";
	private static final String	   	ageSVMname = identRootDir + "/ageSVM.xml";
	
	private static Context   		parentContext	= null;
	
	public expression			   	emotion			= null;
	// labels used in sample categorization
 	public static final int		   	EMOTION_HAPPY = 0,EMOTION_SAD = 1, EMOTION_ANGRY = 2,
 									EMOTION_DISGUSTED = 3, EMOTION_AFRAID = 4,
 								   	EMOTION_NEUTRAL=5; //EMOTION_SURPRISED = 6
 	
 	private final static double s_d = 0.2;
	private final static double m_d = 0.5;
	private final static double l_d = 0.8; // small, medium and large difference values

 	// emotion distances in symmetrical table. indices are from EMOTION_ values above.
	// For example (happy,sad) distance is (0,1) = large distance
 	public static final double[][] exp_dists = {{ 0, l_d, m_d, l_d, l_d, l_d, s_d },
												{ l_d, 0, m_d, m_d, m_d, m_d, s_d },
												{ m_d, m_d, 0, m_d, m_d, m_d, m_d },
												{ l_d, m_d, m_d, 0, m_d, m_d, l_d },
												{ l_d, m_d, m_d, m_d, 0, m_d, l_d },
												{ l_d, m_d, m_d, m_d, m_d, 0, m_d },
												{ s_d, s_d, m_d, l_d, l_d, m_d, 0 }};
 	
 	private CvSVM				   	expressionSVM		= null;
 	private CvSVM				   	genderSVM			= null;
 	private CvSVM					ageSVM				= null;
 	
 	private Mat					   	emotion_histogram	= null;
	private Mat					   	gender_histogram	= null;
	private Mat					   	age_histogram		= null;
	
	public static final int		   	GENDER_MALE = 0, GENDER_FEMALE=1, GENDER_UNKNOWN=2;
	
	public static final int			AGE_YOUNG = 0, AGE_MIDDLEAGED = 1, AGE_OLD = 2;
	
	public static final int			INDEX_IDENTITY=0, INDEX_EXPRESSION=1, INDEX_GENDER=2, INDEX_AGE=3;
	
	static FaceDetectionAndProcessing faceclass 	= null;
	
	public expressionclassifier(Context parent, FaceDetectionAndProcessing parentFaceClass) {
        expressionSVM = new CvSVM();
        expressionSVM.load(expressionSVMname);
        genderSVM = new CvSVM();
        genderSVM.load(genderSVMname);
        ageSVM = new CvSVM();
        ageSVM.load(ageSVMname);
        
        // 3x13 (height x width) kernel gives as good result as 3x15, 9x1 and 9x15
        // according to Naika, Das and Nair, but as it is the smallest of these
        // it is the most efficient to be used.
        emotion = new expression();
        emotion.setFilterSize(13,3);
        
        parentContext = parent;
        faceclass = parentFaceClass;

        // Use all fisherfaces for gender classification, number of eigenfaces set in "faceclass"
        //emotion.initModels(0, 250, faceclass.eigenfaces_saved, 250);
        emotion.initModels(0, 250, 4, 250);
	}
		
	public void skinThreshold( Mat frame, MatOfRect clip, boolean scan ) {
		emotion.skinThreshold( frame, clip, scan );
	}
	
	public void GaborLBP_Histograms( Mat pic, Mat hist, Mat LUT, int N, int step, int ind ) {
		emotion.GaborLBP_Histograms( pic, hist, LUT, N, step, ind );
	}
	
	// Parses the given filename to values used internally
	protected int[] parseAttributes (String fname) {
		Scanner scan = new Scanner(fname).useDelimiter("_");
		
		int id = scan.nextInt();
		String age = scan.next().toLowerCase();
		String gender = scan.next().toLowerCase();
		String emotion = scan.next().toLowerCase();
		
		int[] retval = new int[4];
		retval[INDEX_IDENTITY] = id;
		
		if (age.equals("y")) retval[INDEX_AGE] = AGE_YOUNG;
		if (age.equals("m")) retval[INDEX_AGE] = AGE_MIDDLEAGED;
		if (age.equals("o")) retval[INDEX_AGE] = AGE_OLD;
	
		if (gender.equals("m")) retval[INDEX_GENDER] = GENDER_MALE;
		if (gender.equals("f")) retval[INDEX_GENDER] = GENDER_FEMALE;
		
		if (emotion.equals("a")) retval[INDEX_EXPRESSION] = EMOTION_ANGRY;
		if (emotion.equals("d")) retval[INDEX_EXPRESSION] = EMOTION_DISGUSTED;
		if (emotion.equals("f")) retval[INDEX_EXPRESSION] = EMOTION_AFRAID;
		if (emotion.equals("h")) retval[INDEX_EXPRESSION] = EMOTION_HAPPY;
		if (emotion.equals("s")) retval[INDEX_EXPRESSION] = EMOTION_SAD;
		if (emotion.equals("n")) retval[INDEX_EXPRESSION] = EMOTION_NEUTRAL;
		
		return retval;
	}
	
	public static String expString( int exp ) {
		if (exp == EMOTION_ANGRY) return "angry";
		if (exp == EMOTION_DISGUSTED) return "disgusted";
		if (exp == EMOTION_AFRAID) return "afraid";
		if (exp == EMOTION_HAPPY) return "happy";
		if (exp == EMOTION_SAD) return "sad";
		if (exp == EMOTION_NEUTRAL) return "neutral";
		return "unknown";
	}
		

	// age: http://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=6460367
	public void trainAttributes() {
		if (!identRootDir.exists()) {
			Toast.makeText(parentContext, "Unable to open /PiP while training for attributes",
					Toast.LENGTH_LONG).show();
			return;
		}
		
		File[] files = facesDir.listFiles(new FilenameFilter() {
		    public boolean accept(File dir, String name) {
		        return name.toLowerCase().endsWith(".jpg");
		    }
		});
		
		Mat exp_traindata = new Mat();
		Mat exp_classdata = new Mat();
		
		Mat age_traindata = new Mat();
		Mat age_classdata = new Mat();
		
		Mat gen_traindata = new Mat();
		Mat gen_classdata = new Mat();
		
		Mat iddata = new Mat();
		
		for (File file : files) {
			Bitmap bm = null;
			Mat matpic = new Mat();
			try {
				FileInputStream fis = new FileInputStream(file);
				bm = BitmapFactory.decodeFile(file.getAbsolutePath());
				fis.close();
				Utils.bitmapToMat(bm, matpic);
				Imgproc.cvtColor(matpic, matpic, Imgproc.COLOR_RGB2GRAY);
			} catch (IOException e) {
	            e.printStackTrace();
	            Log.e(TAG, "IOException: " + e);
	        }
			
			MatOfRect MoR = new MatOfRect();
			faceclass.mCascadeClassifier.detectMultiScale(matpic, MoR);
			Rect[] faces = MoR.toArray();
			for (int fi = 0; fi < faces.length; fi++) {
				//skip false positives clearly too small or big to be faces
				float ratio = (float)faces[fi].height / matpic.height();
				if (ratio > 0.4 && ratio < 0.9) {					
					Mat facepic = matpic.submat(faces[fi]).clone();
					// Parse the classes from the filename
					boolean grab = false;
					int[] attrib = parseAttributes( file.getName() );
					
					iddata.push_back(new MatOfInt(attrib[INDEX_IDENTITY]));
					if ((attrib[INDEX_IDENTITY]==140) && (attrib[INDEX_EXPRESSION]==EMOTION_HAPPY)) grab = true;
					
					if (grab) helper.savePicture(facepic, false, "orig");
					
					Imgproc.equalizeHist(facepic, matpic);
					if (grab) helper.savePicture(matpic, false, "eqHist");
										
					/*matpic = faceclass.gammaCorrect(matpic, 2);
					if (grab) helper.savePicture(matpic, false, "gamma");
					
					matpic = faceclass.localNormalization(matpic, 3);
					if (grab) helper.savePicture(matpic, false, "local");*/
					
					// crop off sides for age recognition - they are irrelevant for the task in hand
					Imgproc.resize(matpic, facepic, new Size(60,60));
					Imgproc.resize(matpic, matpic, new Size(64,64));
					
					if (grab) helper.savePicture(facepic, false, "resize");
					
					// expression and gender processing
					Mat exp_hist = new Mat();
					Mat gen_hist = new Mat();
					Mat age_hist = new Mat();
					
					emotion.addFisherface(matpic, attrib[INDEX_GENDER]);

					// Calculate the ARLBP and normal 3x3 LBP using uniform patterns for gender classification
					emotion.ARLBP(matpic, exp_hist, gen_hist, 4, 4);
					// ARLBP alters the size of matpic
					
					exp_traindata.push_back(exp_hist.t().clone());
					exp_classdata.push_back(new MatOfInt(attrib[INDEX_EXPRESSION]));
					
					gen_traindata.push_back(gen_hist.t().clone());				
					gen_classdata.push_back(new MatOfInt(attrib[INDEX_GENDER]));
					
					// age processing				
					facepic = facepic.colRange(5, 53).clone();
										
					Mat means = facepic.clone();
					emotion.ELBP(facepic, 3, 2, 8, (float)Math.PI/2);
					emotion.localMeanThreshold(means, 3, 3);
					
					emotion.concatHist(facepic, means, age_hist);
					//helper.savePicture(facepic, false, "ELBP");
										
					age_traindata.push_back(age_hist.t().clone());
					age_classdata.push_back(new MatOfInt(attrib[INDEX_AGE]));
					
					age_hist = null;
					exp_hist = null;
					gen_hist = null;
					
					continue;
				}
			}
		}
		
		Log.d(TAG, "Training");
		
		age_traindata.convertTo(age_traindata, CvType.CV_32F);
		age_classdata.convertTo(age_classdata, CvType.CV_32F);
		
		exp_traindata.convertTo(exp_traindata, CvType.CV_32F);
		exp_classdata.convertTo(exp_classdata, CvType.CV_32F);
		
		gen_traindata.convertTo(gen_traindata, CvType.CV_32F);
		gen_classdata.convertTo(gen_classdata, CvType.CV_32F);
		
		CvSVMParams params = new CvSVMParams();
		params.set_svm_type(CvSVM.C_SVC);
		params.set_nu(0.5);
		params.set_gamma(1);
		params.set_kernel_type(CvSVM.RBF);
		params.set_term_crit(new TermCriteria(TermCriteria.MAX_ITER + TermCriteria.EPS, 100, 1e-3));
		/*
		if (exp_traindata.size().height > 0) {
			expressionSVM.train_auto(exp_traindata, exp_classdata, new Mat(), new Mat(), params);
			expressionSVM.save(expressionSVMname);
		}		
		
		Log.d(TAG, "Training, expression done.");
		
		if (age_traindata.size().height > 0) {
			ageSVM.train_auto(age_traindata, age_classdata, new Mat(), new Mat(), params);
			ageSVM.save(ageSVMname);
		}
		Log.d(TAG, "Training, age done."); */
		
		if (gen_traindata.size().height > 0) {
			genderSVM.train_auto(gen_traindata, gen_classdata, new Mat(), new Mat(), params);
			genderSVM.save(genderSVMname);
		}
		emotion.trainFisherfaces();
		Log.d(TAG, "Training, gender done.");
		
	}	
	
	// Calculate AR-LBP feature histograms for expression recognition based on
	// "Asymmetric Region Local Binary Pattern Operator for Person-dependent Facial 
	// Expression Recognition" by Naika, Das and Nair. The result is fed to SVM for expression matching.
	// Also 3x3 LBP is calculated and uniform patterns from it are used to categorize targets gender.
	public int[] identifyExpression( Mat matpic ) {
		if (!matpic.size().equals(faceclass.facesize)) Imgproc.resize(matpic, matpic, faceclass.facesize, 0, 0, Imgproc.INTER_AREA);
		int[] retvals = new int[4];
		
		//Log.d(TAG, "matpic:"+matpic);
		
		emotion_histogram = new Mat();
		gender_histogram = new Mat();
		age_histogram = new Mat();
		
		// This first to save processing time, ARLBP changes the size of matpic
		//retvals[INDEX_GENDER] = emotion.predictFisherface(matpic);
		
		helper.savePicture(matpic, false, "in_");
		
		emotion.ARLBP(matpic, emotion_histogram, gender_histogram, 4, 4);
		
		Imgproc.resize(matpic, matpic, new Size(60,60), 0, 0, Imgproc.INTER_AREA);
		
		matpic = matpic.colRange(5, 53).clone();
		Mat means = matpic.clone();
		emotion.ELBP(matpic, 3, 2, 8, (float)Math.PI/2);
		emotion.localMeanThreshold(means, 3, 3);
		emotion.concatHist(matpic, means, age_histogram);
		
		emotion_histogram.convertTo(emotion_histogram, CvType.CV_32F);
		gender_histogram.convertTo(gender_histogram, CvType.CV_32F);
		age_histogram.convertTo(age_histogram, CvType.CV_32F);
		
		retvals[INDEX_EXPRESSION] = (int)expressionSVM.predict(emotion_histogram);
		retvals[INDEX_GENDER] = (int)genderSVM.predict(gender_histogram);
		retvals[INDEX_AGE] = (int)ageSVM.predict(age_histogram);
				
		return retvals;
	}
}
