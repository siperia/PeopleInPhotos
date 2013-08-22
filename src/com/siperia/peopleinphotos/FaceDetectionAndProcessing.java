package com.siperia.peopleinphotos;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.FilenameFilter;
import java.io.IOException;
import java.io.InputStream;
import java.util.AbstractMap;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Scanner;
import java.util.Set;
import java.util.StringTokenizer;
import java.util.Map.Entry;
import java.util.TreeMap;

import org.opencv.android.Utils;
import org.opencv.core.Core;
import org.opencv.core.Core.MinMaxLocResult;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfDouble;
import org.opencv.core.MatOfFloat;
import org.opencv.core.MatOfInt;
import org.opencv.core.MatOfInt4;
import org.opencv.core.MatOfRect;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;
import org.opencv.objdetect.CascadeClassifier;

import com.siperia.peopleinphotos.Identity.sample;

import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.os.Environment;
import android.util.Log;
import android.widget.Toast;

public class FaceDetectionAndProcessing {
	private static final String	   TAG = "FaceDetectionAndProcessing";
	private static final File 	   sdDir = Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_PICTURES);
	private static final File	   identRootDir			= new File(sdDir, "PiP_idents");
	private static final File	   pictureFileDir		= new File(sdDir, "PiP");
	private static final File      facesDir = new File(pictureFileDir, "FACES");
	private static final File      facesTestDir = new File(pictureFileDir, "FACEStest");
	
	public CascadeClassifier	   mCascadeClassifier;
    public DetectionBasedTracker   mNativeDetector;
    private File                   mCascadeFile;
    
    public int                     mDetectorMethod		= VIOLA_JONES;
    public int					   mIdentifierMethod	= EIGENFACES;
    
    public float                   mRelativeFaceSize   = 0.2f;
    public int                     mAbsoluteFaceSize   = 0;
    
 // TODO: gaborit
    //how many features are picked to be used in identification from (facesize*gabor_subpics)
    private static final int	   useableGVLBPfeatures = 120;
    private List<Mat>			   gaborKernels			= null;
    
    private int					   freq_dividers		= 5;
	private int					   rotation_dividers	= 8;
	// stuff used per frame.. initialized here to avoid memory swapping
	private double[] 			   result				= new double[6400];
	private double[]			   partialScore 		= new double[6400];
	private Mat					   LDAProjectionMatrix	= null;
	private MatOfDouble			   GVLBPcandidate		= null;
	private MatOfDouble			   faceSample			= null;
	private MatOfDouble			   selectedSamples		= null;
	// Generic histogram constants
	private static MatOfInt 	   mHistSize			= null;
    private static MatOfFloat 	   mRanges				= null;
        
    private MatOfInt			   LUT					= null;
    // gabor lookup table for histogram reduction
    
    public static final int        VIOLA_JONES			= 0;
    public static final int        LBPBASED				= 1;
    
    public static final int        EIGENFACES			= 0;
    public static final int		   GVLBP				= 1;
    public static final int		   SURF					= 2;
    public static final int		   NO_IDENTIFICATION	= 255;
    
    // Face size and holder for resized sample to be identified by any method
    public final Size			   facesize				= new Size(64,64);
    public final int			   eigenfaces_saved		= 25;
    //private CvSVM			   	   eigenSVM				= null;
    
    private Context				   parentContext;
    
    public ArrayList<Identity>	   identities			= new ArrayList<Identity>();
    
    public expressionclassifier	   expclass				= null;
    
    private double ig = 0;
    MatOfInt gamma_lut = null;
    
    public FaceDetectionAndProcessing(Context parent) { 
    	parentContext = parent;
    	expclass = new expressionclassifier(parent, this);
    	updateSampleFiles();
    	//loadEyeCascades();
    	
    	mHistSize = new MatOfInt(256);
        mRanges = new MatOfFloat(0f, 256f);
        mRanges = new MatOfFloat(mRanges.t());
        
        FileReader fr;
		try {
			fr = new FileReader(new File(pictureFileDir.getAbsolutePath() +"/gabor_lut.txt"));
			BufferedReader br = new BufferedReader(fr);
			Scanner scan = new Scanner( br );
			int i=0;
			int lut[] = new int[256];
			
			while (scan.hasNext()) {
				int in = scan.nextInt();
				lut[i++] = in;
			}
			scan.close(); br.close(); fr.close();
			LUT = new MatOfInt( lut );
			LUT.convertTo(LUT, CvType.CV_8U);
			
		} catch (Exception e) { e.printStackTrace(); }
    }
    
    public void skinThreshold(Mat rgb, MatOfRect clip, boolean scan) {
    	expclass.skinThreshold(rgb, clip, scan);
    }
    
    public int[] identifyFace(Mat facepic)
    {
    	if (!facepic.isContinuous()) facepic = facepic.clone(); // ensures that the matrix is continuous
    	if (facepic.channels() != 1) Imgproc.cvtColor(facepic, facepic, Imgproc.COLOR_RGB2GRAY);
    	
    	//Log.d(TAG, "facepic:"+facepic);
    	    	
    	if (facepic.size().area() < facesize.area()) {
    		Imgproc.resize(facepic, facepic, facesize, 0, 0, Imgproc.INTER_CUBIC);
    	} else if (facepic.size().area() > facesize.area()) {
    		Imgproc.resize(facepic, facepic, facesize, 0, 0, Imgproc.INTER_AREA);
    	} // Slim chance the ROI is actually facesize'd
    	
    	Imgproc.equalizeHist(facepic, facepic);
    	    	
    	int[] retval = expclass.identifyExpression( facepic.clone() );
    	
    	//facepic = gammaCorrect(facepic,2);
    	//facepic = localNormalization(facepic, 3);
    	    	
    	switch (mIdentifierMethod) {
    		case EIGENFACES:
	    	facepic = facepic.reshape(0, 1);
	    	
	    	retval[0] = expclass.emotion.predictEigenface(facepic);
	    	break;
	    	
    	case GVLBP:
    		/*
    		double[] tmpconst = new double[useableGVLBPfeatures];
    		if (selectedSamples == null) selectedSamples = new MatOfDouble();
    		
    		double[] faceSample = calculateGVLBPdouble( facepic );
    		int[] features = identities.get(0).getFeatures();
    		
			for (int i=0;i<tmpconst.length;i++) tmpconst[i] = faceSample[features[i]];
			selectedSamples.fromArray(tmpconst);
			
			Core.gemm(LDAProjectionMatrix, selectedSamples, 1, new Mat(), 0, selectedSamples, Core.GEMM_1_T);
			double d1=Core.norm(selectedSamples, Core.NORM_L2);
			
	    	double best=Double.MAX_VALUE, dist = 0;
			for (Identity ID : identities) {
				for (sample s : ID.getSampleList()) {
					GVLBPcandidate = s.getLDA();
					double d2=Core.norm(GVLBPcandidate, Core.NORM_L2);
					dist = GVLBPcandidate.dot(selectedSamples);					
					
					if ((d1>0) && (d2>0)) dist = dist / (d1*d2);
					//dist = Math.abs(dist);
					
					Log.d(TAG, ID.getName()+" score "+dist);
					
					if (dist<best) {
						best = dist;
						IDstring = ID.getName();
					}
				}
    		}
			Log.d(TAG, "best score: "+best);*/
    		retval[0] = -1;
    		break;
			
    	default: retval[0] = -1; // unknown
    	}
		
		return retval;
    }
    
    public void selectFaceRecognicer(int type) {
    	mDetectorMethod = type;
		try
		{
			if (mNativeDetector != null) {
				mNativeDetector.stop();
				mNativeDetector=null;
			}
			
			int xmlname = 0;			
			File cascadeDir = parentContext.getDir("cascade", Context.MODE_PRIVATE);
			
			switch (mDetectorMethod) {
			case VIOLA_JONES:
				xmlname = R.raw.haarcascade_frontalface_default;
				mCascadeFile = new File(cascadeDir, "haarcascade_frontalface_default.xml");
				break;				
			case LBPBASED:
				xmlname = R.raw.lbpcascade_frontalface;
				mCascadeFile = new File(cascadeDir, "lbpcascade_frontalface.xml");
				break;
			}
			
	    	InputStream is = parentContext.getResources().openRawResource(xmlname);
	        FileOutputStream os = new FileOutputStream(mCascadeFile);
	        byte[] buffer = new byte[4096];
	        int bytesRead;
	        while ((bytesRead = is.read(buffer)) != -1) {
	            os.write(buffer, 0, bytesRead);
	        }
	        is.close();
	        os.close();
	        
	        mNativeDetector = new DetectionBasedTracker(mCascadeFile.getAbsolutePath(), 0);
	        mCascadeClassifier = new CascadeClassifier(mCascadeFile.getAbsolutePath());
	        	        
	        mNativeDetector.setMinFaceSize(64);
	        cascadeDir.delete();
	        
		} catch (IOException e) {
            e.printStackTrace();
            Log.e(TAG, "Failed to load cascade. Exception thrown: " + e);
        }
	}
    
 // Scan the user ident folder and retrain Eigenface classifier with them. 
    public final void trainEigenfaceClassifier() {
    	updateSampleFiles();
    	
    	expclass.trainAttributes();

    	int badcount=0,i=0;
		for (Identity ID : identities) {
			for (sample s : ID.getSampleList()) {
				s.reloadPicture();
				Mat matpic = s.getRawPicture().clone();
				
				// conversion from facesize'd picture to row vector
				if (matpic != null) {
					Log.d(TAG, "matpic:"+matpic);
					Imgproc.resize(matpic, matpic, facesize);
										
					//matpic = gammaCorrect(matpic,2);
			    	//matpic = localNormalization(matpic, 3);
					
					matpic = matpic.reshape(0,1);
					
					expclass.emotion.addEigenface(matpic, ID.getID());
				} else badcount++;
				
				s.releasePicture();
			}
		}
		if (badcount > 0) Toast.makeText(parentContext, "The dataset includes samples where no face could be found. "+badcount,
				Toast.LENGTH_LONG).show();

		expclass.emotion.trainEigenfaces();
		
		
	}
    
    public void saveEigenpictures() { 
    	Mat pic[] = new Mat[5];
		for (int h=0;h<5;h++) pic[h] = Mat.zeros(facesize, CvType.CV_32F);
		
		expclass.emotion.getEigenPictures(pic[0], pic[1], pic[2], pic[3], pic[4]);
		for (int h=0;h<5;h++) {
			Core.normalize(pic[h], pic[h], 0, 255, Core.NORM_MINMAX, CvType.CV_8U);
			pic[h] = pic[h].reshape(0, (int)facesize.width);
		
			helper.savePicture(pic[h], false, "Eigen"+h+"_");
		}
    }
    
    public void updateSampleFiles( ) {
		if (!identRootDir.exists()) {
			Toast.makeText(parentContext, "Unable to open /PiP_idents or a subdirectory",
					Toast.LENGTH_LONG).show();
			return;
		}
		
    	//identities.clear();
		
		for (File child : identRootDir.listFiles()) {
			if (child.isDirectory()) {
				File IDdir = new File( child.getAbsolutePath() );
				
				Identity thisID = null;
				boolean skip = false;
				for (Identity ID: identities) {
					if (ID.equals(IDdir.getName())) {
						thisID = ID;
						skip = true;
					}
					if (skip) break;
				}
				if (!skip) { // new dir, new username
					thisID = new Identity( IDdir.getName(), identities.size() );
					identities.add( thisID );

					// then add files from that directory
					for (File file : IDdir.listFiles()) {
						if (file.isFile() && !file.isDirectory()) {
							//Log.d(TAG, "adding "+file.getAbsolutePath());
							thisID.addSample(file.getAbsolutePath());
						}
					}
				}				
			}
		}
    }
    
    public final List<String> getIdents() {
    	List<String> retval = new ArrayList<String>();
    	for (Identity ID: identities) {
    		retval.add( ID.getName() );
    	}
		return retval;
    }
    
    public Mat gammaCorrect(Mat pic, double gamma) {
		 double inverse_gamma = 1.0 / gamma;
		 
		 if (inverse_gamma != ig) {		 
			 gamma_lut = new MatOfInt(1, 256, CvType.CV_8UC1 );
			 int[] conv = new int[256];
			 for( int i = 0; i < 256; i++ )
			   conv[i] = (int)( Math.pow( (double) i / 255.0, inverse_gamma ) * 255.0 );
			 gamma_lut.fromArray(conv);
			 
			 ig = inverse_gamma;
		 }
		 
		 Core.LUT( pic, gamma_lut, pic );
		 Core.convertScaleAbs(pic, pic);
		 
		 return pic;
	}
    
    public Mat localNormalization(Mat pic, int wsize) {
    	Mat out = new Mat( pic.size(), CvType.CV_32FC1);
    	pic.convertTo(pic,  CvType.CV_32FC1);

    	int side = wsize*2+1;
    	Size winsize = new Size( side,side );
    	Mat diff = new Mat(winsize, CvType.CV_32FC1);
    	Mat var_m = new Mat(winsize, CvType.CV_32FC1);
    	    	
    	Imgproc.copyMakeBorder(pic,pic,wsize,wsize,wsize,wsize, Imgproc.BORDER_REFLECT);
    	Mat mean = new Mat(pic.size(), CvType.CV_32FC1);
    	
    	Imgproc.blur(pic, mean, winsize); //blur = mean box filter

    	for (int y = wsize; y < pic.height()-wsize; y++) {
    		for (int x = wsize; x < pic.width()-wsize; x++) {
    			Mat W = pic.colRange(x-wsize,x+wsize+1).rowRange(y-wsize, y+wsize+1);
    			Mat mW = mean.colRange(x-wsize,x+wsize+1).rowRange(y-wsize, y+wsize+1);

    			Core.subtract(W, mW, diff);
    			Core.pow(diff, 2, var_m);
    			
    			double variance = Math.sqrt( Core.sumElems(var_m).val[0] / winsize.area() ) + 0.01; //avoid 0's
    			double data = diff.get(wsize, wsize)[0] / variance;
    			out.put(y-wsize, x-wsize, data);
    		}
    	}
    	
    	Core.normalize(out, out, 0, 255, Core.NORM_MINMAX, CvType.CV_8U);
    	
    	return out;
    }
	
	// Function to find 8 best mappings for GVLBP-histograms
	public void findSimpleUniformLBPIndices() {
		
		File[] FACESfiles = facesDir.listFiles(new FilenameFilter() {
		    public boolean accept(File dir, String name) {
		        return name.toLowerCase().endsWith(".jpg");
		    }
		});
		Mat sumHist = new Mat();
		mHistSize = new MatOfInt(256);
		mRanges = new MatOfFloat(0f, 256f);
		
		List<Integer> vals = new ArrayList<Integer>();
		boolean first = false;
		if (first) {
			for (int i=0;i<FACESfiles.length;i++) {
				File file = FACESfiles[i];
				
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
				mCascadeClassifier.detectMultiScale(matpic, MoR);
				Rect[] faces = MoR.toArray();
				for (int fi = 0; fi < faces.length; fi++) {
					// skip false positives clearly too small or big to be faces
					float ratio = (float)faces[fi].height / matpic.height();
					
					if (ratio > 0.4 && ratio < 0.9) {
						Mat facepic = matpic.submat(faces[fi]).clone();
						MatOfInt hist = new MatOfInt();
						
						// To use this part change the "normal_operation" to false in native GaborLBPHistograms
						expclass.GaborLBP_Histograms(facepic, hist, new Mat(), 0, 1, 15);
						
						//Core.normalize(facepic, facepic, 0,255,Core.NORM_MINMAX);
						helper.savePicture(facepic, false, "LBP");
						hist.convertTo(hist, CvType.CV_32F);
						
						if (!sumHist.size().equals( hist.size() )) sumHist.create(hist.size(), CvType.CV_32F);
						Core.add(hist,sumHist,sumHist);
						
						Log.d(TAG, "Added "+i+"/"+FACESfiles.length);
					}
				}
			}
			
			// Detailed accuracy isn't the main thing here so this is close enough. 
			Log.d(TAG, "sumHist:"+sumHist);
			//Core.normalize(sumHist, sumHist, 0, 65535, Core.NORM_MINMAX);

			int[] v = new int[1];
			for (int j=0;j<sumHist.height();j++) {
				sumHist.get(j, 0, v);
				vals.add(v[0]);
			}
			
			String prk="";
			for (int h=0;h<vals.size();h++) prk+=Float.toString(vals.get(h)) + " ";
			Log.d(TAG, "list:"+prk);
			
			try {
				FileWriter fw = new FileWriter(new File(pictureFileDir.getAbsolutePath() +"/lbp_counts.txt"), false);
				BufferedWriter bw = new BufferedWriter(fw);
				bw.write(prk);
				bw.close();
				fw.close();
			} catch ( Exception e ) {}
		} else {
			// Collapse the histogram into N largest groups by combining 2 smallest and mark
			// them into the same category until only N categories are remaining
			FileReader fr;
			try {
				fr = new FileReader(new File(pictureFileDir.getAbsolutePath() +"/lbp_counts.txt"));
				BufferedReader br = new BufferedReader(fr);
				Scanner scan = new Scanner( br );
				
				while (scan.hasNext()) {
					int in = (int)scan.nextFloat();
					vals.add( in );
					Log.d(TAG, "lut added "+ in);
				}
				
				scan.close(); br.close(); fr.close();
			} catch (Exception e1) {
				// TODO Auto-generated catch block
				e1.printStackTrace();
			}
			
			int N = 8;
			List<Integer> indToCount = new ArrayList<Integer>();
			TreeMap<Integer, Integer> countToInd = new TreeMap<Integer,Integer>();
			List<Pair<Integer, Integer>> removed = new ArrayList<Pair<Integer,Integer>>();
			
			for (int i=0;i<256;i++) {
				int k = vals.get(i);
				// Remove double values so it can be used in a TreeMap
				boolean ok = true;
				do {
					ok = true;
					for (int e: indToCount) {
						if (e == k) {
							k++;
							ok = false;
							continue;
						}
					}
				} while (!ok);
				
				indToCount.add(k);
				countToInd.put(k,i);
			}
			
			// get 2 smallest values, combine them and remember the replacement connection
			do {
				Entry<Integer, Integer> smallestEntry = countToInd.pollFirstEntry();
				Entry<Integer, Integer> secondEntry = countToInd.pollFirstEntry();
				int sum = smallestEntry.getKey() + secondEntry.getKey();
				
				while (countToInd.containsKey(sum)) sum++;
								
				countToInd.put( sum, secondEntry.getValue() );
				
				removed.add(new Pair<Integer,Integer>(secondEntry.getValue(), smallestEntry.getValue()));
				
				// update already removed entries to the new key
				for (Pair<Integer,Integer> e: removed) {
					if (smallestEntry.getValue() == e.getFirst()) e.setFirst(secondEntry.getValue());
				}				
				
				Log.d(TAG, "lut: i2c:"+indToCount.size()+", c2i:"+countToInd.size()+", removed:"+removed.size());
			} while (countToInd.size() > N);
			
			// Construct a Look-up-table
			// First, remaining N largest
			int lut[] = new int[256];
			for(Map.Entry<Integer,Integer> e : countToInd.entrySet()) {
				lut[e.getValue()] = e.getValue();
			}
			Log.d(TAG, "lut:"+new MatOfInt(lut).dump() );
			
			// and the removed smaller ones.
			for ( Pair<Integer,Integer> e : removed ) {
				lut[ e.getSecond() ] = e.getFirst();
			}
			
			// convert the values to range 0-N here - I did it outside the program
			
			Log.d(TAG, "lut:"+new MatOfInt(lut).dump() );
			try {
				String s ="";
				for (int i = 0;i < 256; i++) s+= Integer.toString(lut[i]) + " ";
			
				FileWriter fw = new FileWriter(new File(pictureFileDir.getAbsolutePath() +"/gabor_lut.txt"), false);
				BufferedWriter bw = new BufferedWriter(fw);
				bw.write(s);
				bw.close();
				fw.close();
			} catch ( Exception e ) {}
		}
	}
    
	public void genderConfusion() {
		// Calculate confusion matrix for FACES with current gender classifier
		
		int[][] confusion = new int[3][3];
		
		File[] FACEStestfiles = facesTestDir.listFiles(new FilenameFilter() {
		    public boolean accept(File dir, String name) {
		        return name.toLowerCase().endsWith(".jpg");
		    }
		});
		
		for (File file : FACEStestfiles) {
			int[] attr = expclass.parseAttributes(file.getName());
			
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
			mCascadeClassifier.detectMultiScale(matpic, MoR);
			Rect[] faces = MoR.toArray();
			for (int fi = 0; fi < faces.length; fi++) {
				float ratio = (float)faces[fi].height / matpic.height();
				if (ratio > 0.4 && ratio < 0.9) {					
					Mat facepic = matpic.submat(faces[fi]).clone();
				
					int[] recatt = expclass.identifyExpression(facepic);
					
					confusion[attr[expressionclassifier.INDEX_GENDER]][recatt[expressionclassifier.INDEX_GENDER]]++;
					
					continue;
				}
			}
		}
		try {
			FileWriter fw = new FileWriter(new File(pictureFileDir.getAbsolutePath() +"/gender_confusion.txt"), false);
			BufferedWriter bw = new BufferedWriter(fw);
			for (int i=0;i<3;i++) {
				String s ="";
				for (int j=0;j<3;j++) s += Integer.toString(confusion[i][j])+" ";
				bw.write(s); bw.newLine();
			}
			bw.close();
			fw.close();
		} catch ( Exception e ) {}
		
	}
	
    // Feeds every sample in PiP_idents subfolders and builds a set to compare against	
    public void trainGVLBPClassifier() {
    	if (identities.isEmpty()) updateSampleFiles();

    	//findSimpleUniformLBPIndices(); //used to create the LUT
    	int histlen = 0;
		for (Identity ID: identities) {
			for (sample s: ID.getSampleList() ) {				
				Mat hist = new Mat();
				s.reloadPicture();
				Mat pic = s.getRawPicture();
				
				if (pic != null) expclass.GaborLBP_Histograms(pic, hist, LUT, 8, -1, -1);
				s.setGaborMat(hist);
				s.releasePicture();
				
				Log.d(TAG, "HIST="+hist.size()+" : "+hist.dump());
				histlen = (int)hist.size().width; // same for all of them
			}
		}
		
		int totalSamples = 0;
		for (Identity ID : identities) totalSamples += ID.getNumberOfSamples();
		// classneed is the total number of picture combinations. The required length for the
		// classification vector.
		int index = 0, classneed=0;
		for (Identity ID : identities)
		for (Identity inID : identities)
		if (inID.getID() >= ID.getID())
		for (@SuppressWarnings("unused") sample s : ID.getSampleList())
		for (@SuppressWarnings("unused") sample in_s : inID.getSampleList())
			classneed++;
		
		List<float[]> allData = new ArrayList<float[]>();
		double[] classVector = new double[ classneed ];
		float[] tn = new float[6400];
		MatOfFloat diff = new MatOfFloat();
		index = 0;
		Log.d(TAG, "hopasd 2");
		for (Identity ID : identities) {
			for (Identity inID : identities) {
				if (inID.getID() >= ID.getID()) {
					// Every inter-pair processed only once
					Log.d(TAG, "hopasd 3");
					for (sample s : ID.getSampleList()) {
						for (sample in_s : inID.getSampleList()) {
							// create should do any memory operations only on the first pass
							diff.create(s.getGaborMat().size(), CvType.CV_32F);
							Core.absdiff(s.getGaborMat(), in_s.getGaborMat(), diff);
							diff.convertTo(diff, CvType.CV_32F);
							allData.add(diff.toArray().clone());
							
							if (inID.getID() == ID.getID()) {
								classVector[index] = 1;
							} else {
								classVector[index] = -1;
							}
							//Log.d(TAG, "index "+index+", inID:"+inID.getID()+", ID:"+ID.getID()+" >> class "+classVector[index]);
							index++;
						}
					}
				}
				// skip allready done cases (inID < ID)
			}
		}
		diff = null;
		Log.d(TAG, "hopasd 5");
		// the data needs to be binarized for MI and CMI, so lets calculate the
		// best possible thresholds for each feature.
		tn=calculateBinarizationThresholds(allData, classVector);
		// and binarize allData for feature selection
		for (float[] fl: allData) {
			for (int i=0;i<tn.length;i++) {
				if (fl[i] > tn[i]) fl[i] = 1; else fl[i] = 0;
			}
		}
		Log.d(TAG, "hopasd 6");
		int[] selectedFeatures = selectFeaturesByMutualInformation(useableGVLBPfeatures, allData, classVector);
		//Log.d(TAG,"selected:"+ItoString(selectedFeatures));

		for (Identity ID: identities) {
			ID.setFeaturesAndReduce(selectedFeatures);
			//for (sample s : ID.getSampleList()) Log.d(TAG, "sF:"+s.getFilteredGabor().dump());
		}
		
		// Now we have the per sample feature vectors all nice and tight in classes
		// In order to speed up the identification and to increase the class separation the
		// next step is to calculate a projection matrix with Linear Discriminant Analysis (LDA)
				
		// Calculate per class and global means
		Mat meanFace = new Mat();
		Mat globalGaborMean = new Mat();
		meanFace = Mat.zeros(new Size(1,selectedFeatures.length), CvType.CV_64F);
		for (Identity ID: identities) {
			ID.calculateMeanGabor();
			for (sample s : ID.getSampleList()) {
				Core.add(s.getFilteredGabor(), meanFace, meanFace);
			}
		}
		Core.divide(meanFace, Scalar.all(totalSamples), globalGaborMean);
		Log.d(TAG, "globalGaborMean:"+globalGaborMean.size()+", "+globalGaborMean.dump());
		meanFace = null;
		
		Mat delta = new Mat();
		Mat scatter = new Mat();
		Mat withinClassScatterSum = new Mat();
		Mat betweenClassScatterSum = new Mat();
		Mat classScatter = new Mat();
		
		betweenClassScatterSum = Mat.zeros(new Size(useableGVLBPfeatures, useableGVLBPfeatures), CvType.CV_64F);
		withinClassScatterSum = Mat.zeros(new Size(useableGVLBPfeatures,useableGVLBPfeatures), CvType.CV_64F);
		delta = Mat.zeros(new Size(1,globalGaborMean.height()), CvType.CV_64F);
		scatter = Mat.zeros(new Size(delta.height(), delta.height()), CvType.CV_64F);
		
		for (Identity ID: identities) {
			Core.subtract(ID.getMeanVector(), globalGaborMean, delta);
			// gemm=general matrix multiplication, transpose 2nd
			Core.gemm(delta, delta, 1, new Mat(), 0, scatter, Core.GEMM_2_T);
			Core.multiply(scatter, Scalar.all(ID.getNumberOfSamples()), scatter);
			Core.add(scatter, betweenClassScatterSum, betweenClassScatterSum);
			
			if (classScatter == null) classScatter = new Mat();
			for (sample s: ID.getSampleList()) {
				classScatter = Mat.zeros(classScatter.size(), classScatter.type());
				Core.subtract(s.getFilteredGabor(),ID.getMeanVector(), classScatter);
				//Log.d(TAG, "cS line:"+classScatter.dump());
				
				Core.gemm(classScatter, classScatter, 1, new Mat(), 0, scatter, Core.GEMM_2_T);
				Core.add(scatter, withinClassScatterSum, withinClassScatterSum);
			}
		}
		delta = null;
		Core.divide(betweenClassScatterSum, Scalar.all(totalSamples), betweenClassScatterSum);
		Core.divide(withinClassScatterSum, Scalar.all(totalSamples), withinClassScatterSum);
		
		// Solve the eigenvalue problem (Sb^-1 Sw)W = W labda
		Core.gemm( withinClassScatterSum.inv(Core.DECOMP_SVD),
				betweenClassScatterSum, 1, new Mat(), 0, scatter, 0);
		//Log.d(TAG, "LDAs:"+scatter.size()+", con: "+scatter.dump());
		LDAProjectionMatrix = new MatOfDouble();
		LDAProjectionMatrix.create(scatter.size(), CvType.CV_64F);
		Mat eigenvalues = new Mat();
		Core.eigen(scatter, true, eigenvalues, LDAProjectionMatrix);
		
		for (Identity ID: identities) {
			ID.updateToLDA( LDAProjectionMatrix );
			//for (sample s: ID.getSampleList()) Log.d(TAG, "ID:"+ID.getName()+" LDA:" + s.getLDA().dump());
		}
		scatter = null;
	}
      
	private int[] selectFeaturesByMutualInformation(int k, List<float[]> pdist, double[] classVector) {
    	assert( classVector.length > 0 );
    	assert( pdist.size() > 0 );
    	assert( pdist.size() == classVector.length );
    	
    	int noOfFeatures = pdist.get(0).length;
    	
    	Arrays.fill(partialScore, 0);
    	
    	int[] m = new int[noOfFeatures];
    	int[] answerFeatures = new int[useableGVLBPfeatures];
    	int highestMICounter = 0, n=0;
    	double highestMI = 0, score = 0, conditionalInfo=0;
    	double[] feature_k = new double[pdist.size()];
    	double[] feature_column = new double[pdist.size()];
    	
    	for (n=0;n<noOfFeatures;n++) {
    		for (int j=0;j<pdist.size();j++) feature_k[j] = (double)pdist.get(j)[n];
			partialScore[n] = JavaMI.MutualInformation.calculateMutualInformation( feature_k, classVector );
			if (partialScore[n] > highestMI) {
				highestMI = partialScore[n];
				highestMICounter = n;
			}
    	}
    	Arrays.fill(answerFeatures, -1);
    	answerFeatures[0] = highestMICounter;
    	for (int i=1;i<k;i++) {
    		score = 0;
    		int limitI = i - 1;
    		for( n=0; n<noOfFeatures; n++) {
    			while ((partialScore[n] >= score) && (m[n] < limitI)) {
    				m[n] = m[n] + 1;
    				for (int u=0;u<pdist.size();u++) {
    					feature_column[u] = pdist.get(u)[answerFeatures[m[n]]];
    					feature_k[u] = pdist.get(u)[n];
    				}
					conditionalInfo = JavaMI.MutualInformation.calculateConditionalMutualInformation(
							classVector,feature_column,feature_k );
					if (partialScore[n] > conditionalInfo) {
						partialScore[n] = conditionalInfo;
					}
    			}
	    		if (partialScore[n] >= score && (answerFeatures[0] != n)) {
	    			score = partialScore[n];
	    			answerFeatures[i] = n;
	    		}
    		}
    	}
    	return answerFeatures;
	}

    
    // calculates the threshold for limiting feature values to binary in GVLBP
    // eq.12 in the relevant document
    private float[] calculateBinarizationThresholds(List<float[]> pdist, double[] classVector) {
    	assert(pdist.size() == classVector.length);
    	
    	float[] ret_best_tn = new float[ pdist.get(0).length ];
    	for (int feature = 0; feature < pdist.get(0).length; feature++) {
    	    	// how many threshold values are attempted between min(pdist) and max(pdist)
	    	final int steps = 16;
	    	float f=0,g=Float.MAX_VALUE;
	    	int i = 0;
	    	while (i < pdist.size()) {
	    		if (pdist.get(i)[feature] > f) f = pdist.get(i)[feature];
	    		if (pdist.get(i)[feature] < g) g = pdist.get(i)[feature];
	    		i++;
	    	}
	    	float best = Float.MAX_VALUE, best_tn= Float.MAX_VALUE, searchstep = (f-g)/steps;
	    	if ((f-g) > 0) {
		    	for (float tn=g; tn<=f; tn+=searchstep) {
			    	float sum=0;
			    	for (i=0;i<pdist.size();i++) {
			    		if (classVector[i] > 0) { //inter sample
			    			if (pdist.get(i)[feature] >= tn) sum += pdist.get(i)[feature]; 
			    		} else { // intra sample
			    			if (pdist.get(i)[feature] < tn) sum += pdist.get(i)[feature];
			    		}
			    	}
			    	if (sum < best) {
			    		best = sum;
			    		best_tn = tn;
			    	}
		    	}
		    	ret_best_tn[feature] = best_tn;
	    	}
    	}
    	return ret_best_tn;
    }
    
}
