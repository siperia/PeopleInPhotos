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
import java.util.Random;
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
import org.opencv.core.TermCriteria;
import org.opencv.imgproc.Imgproc;
import org.opencv.ml.CvSVM;
import org.opencv.ml.CvSVMParams;
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
	protected static final File	   sdDir = Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_PICTURES);
	protected static final File	   identRootDir			= new File(sdDir, "PiP_idents");
	protected static final File	   pictureFileDir		= new File(sdDir, "PiP");
	protected static final File    facesDir = new File(pictureFileDir, "FACES");
	protected static final File    facesTestDir = new File(pictureFileDir, "FACEStest");
	
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
    private MatOfDouble			   LDAProjectionMatrix  = null;
        
	private Mat					   GVLBPcandidate		= null;
	private MatOfDouble			   faceSample			= null;
	private MatOfDouble			   selectedSamples		= null;
	
	private CvSVM				   gaborSVM				= null;
	private final double		   GVLBPReject			= 0.89;

	// Generic histogram constants	
    private static MatOfInt 	   mHistSize			= null;
	private static MatOfFloat 	   mRanges				= null;
        
    protected MatOfInt			   LUT					= null;
    // gabor lookup table for histogram reduction
    
    public static final int        VIOLA_JONES			= 0;
    public static final int        LBPBASED				= 1;
    
    public static final int        EIGENFACES			= 0;
    public static final int		   GVLBP				= 1;
    public static final int		   SURF					= 2;
    public static final int		   NO_IDENTIFICATION	= 255;
    
    // Face size and holder for resized sample to be identified by any method
    public final Size			   facesize				= new Size(64,64);
    public final int			   eigenfaces_saved		= 40;
        
    public final int			   eigen_threshold		= 300;
    // allowed error distance for the recognized
    
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
    	expclass.nativelib.skinThreshold(rgb, clip, scan);
    }
    
    public int[] identifyFace(Mat facepic)
    {
    	if (!facepic.isContinuous()) facepic = facepic.clone(); // ensures that the matrix is continuous
    	    	    	
    	if (facepic.size().area() < facesize.area()) {
    		Imgproc.resize(facepic, facepic, facesize, 0, 0, Imgproc.INTER_CUBIC);
    	} else if (facepic.size().area() > facesize.area()) {
    		Imgproc.resize(facepic, facepic, facesize, 0, 0, Imgproc.INTER_AREA);
    	} // Slim chance the ROI is actually facesize'd
    	
    	if (facepic.channels() != 1) Imgproc.cvtColor(facepic, facepic, Imgproc.COLOR_RGB2GRAY);
    	Imgproc.equalizeHist(facepic, facepic);
    	    	
    	// TODO, facetesting !
    	int[] retval = expclass.identifyExpression( facepic.clone() );
    	//int[] retval = new int[4];
    	
    	//facepic = gammaCorrect(facepic,2);
    	//facepic = localNormalization(facepic, 3);
    	
    	switch (mIdentifierMethod) {
    		case EIGENFACES:
		    	facepic = facepic.reshape(0, 1);
		    	
		    	retval[0] = expclass.nativelib.predictEigenface(facepic);
		    	break;
		    	
	    	case GVLBP:
	    		GVLBPcandidate = new Mat();
	    		
	    		/*expclass.localMeanThreshold(facepic, 8, 8);
				helper.savePicture(facepic, false, "local");*/
	    		
	    		expclass.nativelib.GaborLBP_Histograms(facepic, GVLBPcandidate, LUT, 8, -1,-1);
	    		
	    		Identity tryID = null;
	    		retval[0] = (int)gaborSVM.predict(GVLBPcandidate.t());
	    		// SVM returns some wild ideas as a match if the face is unknown.. lets template match this ID
	    		// with known samples for that ID and reject if the histogram distance is too much
	    		for (Identity ID: identities) if (ID.getID()==retval[0]) tryID = ID;
	    		if (tryID != null) {
	    			double max = 0;
	    			double match = 0;
	    			for (sample s: tryID.getSampleList()) {
	    				match = Imgproc.compareHist(GVLBPcandidate, s.getGaborMat(), Imgproc.CV_COMP_CORREL);
	    				if (match > max) max = match;
	    			}
	    			Log.d(TAG, "debug: maxmatch:"+max);
	    			if (max < GVLBPReject) retval[0] = -1;
	    		}
	    		
	    		
			/*Core.gemm(LDAProjectionMatrix, selectedSamples, 1, new Mat(), 0, selectedSamples, Core.GEMM_1_T);
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
	        
	        mNativeDetector = new DetectionBasedTracker(mCascadeFile.getAbsolutePath(), 32);
	        	        
	        mCascadeClassifier = new CascadeClassifier(mCascadeFile.getAbsolutePath());
	        
	        cascadeDir.delete();
	        
		} catch (IOException e) {
            e.printStackTrace();
            Log.e(TAG, "Failed to load cascade. Exception thrown: " + e);
        }
	}
    
 // Scan the user ident folder and retrain Eigenface classifier with them. 
    public final void trainEigenfaceClassifier() {
    	updateSampleFiles();
   	
    	expclass.nativelib.initModels(0, expressionclassifier.fisher_threshold, eigenfaces_saved, eigen_threshold, false);

    	int badcount=0,i=0;
		for (Identity ID : identities) {
			for (sample s : ID.getSampleList()) {
				s.reloadPicture();
				Mat matpic = s.getRawPicture().clone();
				
				// conversion from facesize'd picture to row vector
				if (matpic != null) {
					Imgproc.resize(matpic, matpic, facesize);
										
					//matpic = gammaCorrect(matpic,2);
			    	//matpic = localNormalization(matpic, 3);
					
					Mat tmp = matpic.clone();
					matpic = matpic.reshape(0,1);
					expclass.nativelib.addEigenface(matpic, ID.getID());
					Core.flip(tmp, matpic, 1);
					matpic = matpic.reshape(0,1);
					expclass.nativelib.addEigenface(matpic, ID.getID());
					
					/*matpic = matpic.reshape(0,64);
					helper.savePair(tmp,  matpic, "pair");*/
					
				} else badcount++;
				
				s.releasePicture();
			}
		}
		if (badcount > 0) Toast.makeText(parentContext, "The dataset includes samples where no face could be found. "+badcount,
				Toast.LENGTH_LONG).show();

		expclass.nativelib.trainEigenfaces();
	}
    
    public void saveEigenpictures() { 
    	Mat pic[] = new Mat[5];
		for (int h=0;h<5;h++) pic[h] = Mat.zeros(facesize, CvType.CV_32F);
		
		expclass.nativelib.getEigenPictures(pic[0], pic[1], pic[2], pic[3], pic[4]);
		for (int h=0;h<5;h++) {
			Core.normalize(pic[h], pic[h], 0, 255, Core.NORM_MINMAX, CvType.CV_8U);
			pic[h] = pic[h].reshape(0, (int)facesize.width);
		
			helper.savePicture(pic[h], null, false, "Eigen"+h+"_");
		}
    }
    
    public void updateSampleFiles( ) {
		if (!identRootDir.exists()) {
			Toast.makeText(parentContext, "Unable to open /PiP_idents or a subdirectory",
					Toast.LENGTH_LONG).show();
			return;
		}
		    	
		File[] idpics = getJPGList( identRootDir );
		if (idpics == null) return;
		identities.clear();
		
		for (File pic : idpics) {
			Scanner scan = new Scanner(pic.getName()).useDelimiter("_");
			
			String name = scan.next();
			int id = scan.nextInt();
			
			Identity thisID = null;
			for (Identity ID: identities) {
				if (ID.getName().equals(name)) {
					thisID = ID;
					continue;
				}
			}
			
			if (thisID == null) {
				thisID = new Identity( name, id );
				identities.add( thisID );
				Log.d(TAG, "debug: added "+name+", "+id);
			}
			
			thisID.addSample(pic.getAbsolutePath());
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
		
		File[] FACESfiles = getJPGList(facesDir);
		
		Mat sumHist = new Mat();
		mHistSize = new MatOfInt(256);
		mRanges = new MatOfFloat(0f, 256f);
		
		List<Integer> vals = new ArrayList<Integer>();
		boolean first = false;
		if (first) {
			for (int i=0;i<FACESfiles.length;i++) {
				File file = FACESfiles[i];
				
				Mat facepic = getFace(file);
				MatOfInt hist = new MatOfInt();
				
				// To use this part change the "normal_operation" to false in native GaborLBPHistograms
				expclass.nativelib.GaborLBP_Histograms(facepic, hist, new Mat(), 0, 1, 15);
				
				//Core.normalize(facepic, facepic, 0,255,Core.NORM_MINMAX);
				/*helper.savePicture(facepic, false, "LBP");
				hist.convertTo(hist, CvType.CV_32F);*/
				
				if (!sumHist.size().equals( hist.size() )) sumHist.create(hist.size(), CvType.CV_32F);
				Core.add(hist,sumHist,sumHist);
				
				Log.d(TAG, "Added "+i+"/"+FACESfiles.length);
					
			}
			
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
    
	public void confusionMatrices() {
		// Calculate confusion matrices for FACES
		
		int[][] gen_confusion = new int[3][3];
		int[][] age_confusion = new int[4][4];
		int[][] emo_confusion = new int[7][7];
		int badcount[] = new int[4];
		
		File[] FACEStestfiles = getJPGList(facesTestDir);
		
		for (File file : FACEStestfiles) {
			int[] attr = expclass.parseAttributes(file.getName());
								
			Mat facepic = getFace(file);
		
			int[] recatt = expclass.identifyExpression(facepic);
			
			/*String s="female";
			if ( recatt[expressionclassifier.INDEX_GENDER] == expressionclassifier.GENDER_MALE) s = "male";
			helper.savePicture(facepic, false, "recatt_"+s);*/
			
			if (recatt[expressionclassifier.INDEX_GENDER] >= 0) {
				gen_confusion[attr[expressionclassifier.INDEX_GENDER]][recatt[expressionclassifier.INDEX_GENDER]]++;
			} else badcount[expressionclassifier.INDEX_GENDER]++;
			
			if (recatt[expressionclassifier.INDEX_AGE] >= 0) {
				age_confusion[attr[expressionclassifier.INDEX_AGE]][recatt[expressionclassifier.INDEX_AGE]]++;
			} else badcount[expressionclassifier.INDEX_AGE]++;
				
			if (recatt[expressionclassifier.INDEX_EXPRESSION] >= 0) {
				emo_confusion[attr[expressionclassifier.INDEX_EXPRESSION]][recatt[expressionclassifier.INDEX_EXPRESSION]]++;
			} else badcount[expressionclassifier.INDEX_EXPRESSION]++;
		}
		try {
			FileWriter fw = new FileWriter(new File(pictureFileDir.getAbsolutePath() +"/confusions.txt"), false);
			BufferedWriter bw = new BufferedWriter(fw);
			for (int i=0;i<3;i++) {
				String s ="";
				for (int j=0;j<3;j++) s += Integer.toString(gen_confusion[i][j])+" ";
				bw.write(s); bw.newLine();
			}
			bw.newLine();
			
			for (int i=0;i<4;i++) {
				String s ="";
				for (int j=0;j<4;j++) s += Integer.toString(age_confusion[i][j])+" ";
				bw.write(s); bw.newLine();
			}
			bw.newLine();
			
			for (int i=0;i<7;i++) {
				String s ="";
				for (int j=0;j<7;j++) s += Integer.toString(emo_confusion[i][j])+" ";
				bw.write(s); bw.newLine();
			}
			bw.newLine();
			
			String s ="";
			for (int j=0;j<4;j++) s += Integer.toString(badcount[j])+" ";
			
			bw.write(s); bw.newLine();
			
			bw.close();
			fw.close();
		} catch ( Exception e ) { helper.crash(); }
		
	}
	
    // Feeds every sample in PiP_idents subfolders and builds a set to compare against	
    public void trainGVLBPClassifier() {
    	updateSampleFiles();

    	//findSimpleUniformLBPIndices(); //used to create the LUT
    	int histlen = 0, totalSamples = 0, index = 0;
    	
    	Mat training = new Mat();
    	Mat classes = new Mat();
    	
		for (Identity ID: identities) {
			for (sample s: ID.getSampleList() ) {
				Mat hist = new Mat();
				s.reloadPicture();
				Mat pic = s.getRawPicture();
				
				if (s.getRawPicture() != null) {
					Imgproc.resize(pic, pic, new Size(64,64));
					Imgproc.equalizeHist(pic, pic);
					
					//expclass.localMeanThreshold(pic, 8, 8);
					expclass.nativelib.GaborLBP_Histograms(pic, hist, LUT, 8, -1,-1);
					
					training.push_back(hist.t().clone());
					classes.push_back(new MatOfInt(ID.getID()));
					
					//Log.d(TAG, "debug hist:"+ hist.dump());
					
					s.setGaborMat(hist);
					s.releasePicture();
				}
								
				histlen = (int)hist.size().height; // same for all of them
				totalSamples++;
			}
		}
		
		Log.d(TAG, "debug data:"+training.size()+", classes:"+classes.size());
		
		CvSVMParams cvparams = new CvSVMParams();
		cvparams.set_kernel_type( CvSVM.RBF );
		cvparams.set_svm_type(CvSVM.C_SVC);
		cvparams.set_C(100);
		cvparams.set_gamma(0.01);
		//cvparams.set_C(1);
		cvparams.set_term_crit(new TermCriteria(TermCriteria.MAX_ITER, 10000, 0.00001));
		gaborSVM = new CvSVM(training, classes, new Mat(), new Mat(), cvparams);
						
		//for (Identity ID : identities) totalSamples += ID.getNumberOfSamples();
		// classneed is the total number of picture combinations. The required length for the
		// classification vector.
		
		//Log.d(TAG, "debug: classneed "+classneed);
		
		/*
		List<float[]> allData = new ArrayList<float[]>();
		List<Double> classList = new ArrayList<Double>();
		float[] tn = new float[histlen];
		MatOfFloat diff = new MatOfFloat();
		index = 0;
		int ones=0, minuses=0;
		
		for (Identity ID : identities) {
			for (Identity inID : identities) {
				if (inID.getID() >= ID.getID()) {
					// Every inter-pair processed only once
					for (sample s : ID.getSampleList()) {
						for (sample in_s : inID.getSampleList()) {
							// create should do any memory operations only on the first pass
							diff.create(s.getGaborMat().size(), CvType.CV_32F);
							
							if ( !s.getGaborMat().equals( in_s.getGaborMat() ) ) {
							
								Core.absdiff(s.getGaborMat(), in_s.getGaborMat(), diff);
								diff.convertTo(diff, CvType.CV_32F);
								allData.add(diff.toArray().clone());
								
								Log.d(TAG, "debug: "+ s.getFileName() +" - " + in_s.getFileName() +" - "+ diff.dump());
								
								if (inID.getID() == ID.getID()) {
									classList.add(1.0); ones++;
								} else {
									classList.add(-1.0); minuses++;
								}

								index++;
							}
						}
					}
				}
				// skip allready done cases (inID < ID)
			}
		}
		diff = null;
		//Log.d(TAG, "debug dump: " + helper.FtoString( allData.get(2)) );
		
		double[] classVector = new double[ classList.size() ];
		for (int h=0;h<classList.size();h++) classVector[h] = classList.get(h); // ...
		
		// the data needs to be binarized for MI and CMI, so lets calculate the
		// best possible thresholds for each feature.
		tn=calculateBinarizationThresholds(allData, classVector);
		// and binarize allData for feature selection
		Log.d(TAG, "debug CV: " + helper.DtoString(classVector));
		Log.d(TAG, "debug feature limits: " + helper.FtoString(tn));
				
		ones = 0; minuses = 0;
		for (float[] fl: allData) {
			for (int i=0;i<tn.length;i++) {
				if (fl[i] >= tn[i]) fl[i] = 1;
				else fl[i] = 0;
			}
		}
		
		Log.d(TAG, "debug binarization, ones - zeroes :"+ones+" - "+minuses);
		Log.d(TAG, "debug dump: " + helper.FtoString( allData.get(2)) );
		
		int[] selectedFeatures = selectFeaturesByMutualInformation(useableGVLBPfeatures, allData, classVector);
		Log.d(TAG,"debug selected:"+helper.ItoString(selectedFeatures));

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
		for (Identity ID: identities) {
			ID.calculateMeanGabor();
			for (sample s : ID.getSampleList()) {
				if (!meanFace.size().equals( s.getFilteredGabor().size()))
					meanFace.create(s.getFilteredGabor().size(), s.getFilteredGabor().type());

				Core.add(s.getFilteredGabor(), meanFace, meanFace);
				Log.d(TAG, "debug filtered :"+s.getFilteredGabor().dump());
			}
		}
		Core.divide(meanFace, Scalar.all(totalSamples), globalGaborMean);
		Log.d(TAG, "debug globalGaborMean:"+globalGaborMean.size()+", "+globalGaborMean.dump());
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
		
		Log.d(TAG, "debug wCSS: "+withinClassScatterSum.dump());
		Log.d(TAG, "debug bCSS: "+betweenClassScatterSum.dump());
				
		// Solve the eigenvalue problem (Sb^-1 Sw)W = W lambda
		Core.gemm( withinClassScatterSum.inv(Core.DECOMP_SVD),
				   betweenClassScatterSum, 1, new Mat(), 0, scatter, 0);
		//Log.d(TAG, "LDAs:"+scatter.size()+", con: "+scatter.dump());
		LDAProjectionMatrix = new MatOfDouble();
		LDAProjectionMatrix.create(scatter.size(), CvType.CV_64F);
		Mat eigenvalues = new Mat();
		Core.eigen(scatter, true, eigenvalues, LDAProjectionMatrix);
		
		Log.d(TAG, "debug projection: "+LDAProjectionMatrix);
		
		for (Identity ID: identities) {
			ID.updateToLDA( LDAProjectionMatrix );
			//for (sample s: ID.getSampleList()) Log.d(TAG, "ID:"+ID.getName()+" LDA:" + s.getLDA().dump());
		}
		scatter = null;
		*/
	}
    
	private int[] selectFeaturesByMutualInformation(int k, List<float[]> pdist, double[] classVector) {
    	assert( classVector.length > 0 );
    	assert( pdist.size() > 0 );
    	assert( pdist.size() == classVector.length );
    	
    	int noOfFeatures = pdist.get(0).length;
    	
    	double partialScore[] = new double[ noOfFeatures ];
    	//Arrays.fill(partialScore, 0);
    	
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
    	assert(classVector.length > 0);
    	assert(pdist.size() == classVector.length);
    	
    	int features = pdist.get(0).length;
    	int samples = pdist.size();
    	
    	float[] ret_best_tn = new float[ features ];
    	for (int feature = 0; feature < features; feature++) {
	    	// how many threshold values are attempted between min(pdist) and max(pdist)
	    	final int steps = 128;
	    	float maxim=0,minim=Float.MAX_VALUE;
	    	
	    	for (int i = 0; i < samples; i++) {
	    		if (pdist.get(i)[feature] > maxim) maxim = pdist.get(i)[feature];
	    		if (pdist.get(i)[feature] < minim) minim = pdist.get(i)[feature];
	    	}
	    	
	    	float best = Float.MAX_VALUE, best_tn= Float.MAX_VALUE, searchstep = (maxim-minim)/steps;
	    	//Log.d(TAG, "debug ss:"+searchstep);
	    	float sum=0;
	    	if ((maxim-minim) > 0) {
		    	for (float tn=minim; tn<=maxim; tn+=searchstep) {
			    	sum = 0;
			    	for (int i=0;i<samples;i++) {
			    		if (classVector[i] > 0) {
			    			// inter-sample
			    			if (pdist.get(i)[feature] >= tn) { sum += pdist.get(i)[feature]; }
			    		} else {
			    			// intra-sample
			    			if (pdist.get(i)[feature] < tn) { sum += pdist.get(i)[feature]; }
			    		}
			    	}
			    	//Log.d(TAG, "debug sum:"+sum);
			    	if (sum < best) {
			    		best = sum;
			    		best_tn = tn;
			    		//Log.d(TAG, "debug best:"+best);
			    	}
		    	}
		    	ret_best_tn[feature] = best_tn;
		    	//Log.d(TAG, "debug tn:"+best_tn);
	    	}
    	}
    	return ret_best_tn;
    }

    // test recognition accuracy with currently selected method
	public void recognitionTest() {
		Random generator = new Random( System.currentTimeMillis() );
		File[] FACESfiles = getJPGList(facesTestDir);
		final String resultFile = identRootDir.getAbsolutePath() + "/facerecog_result.txt";
		
		Map<Integer,Integer> idlist = new TreeMap<Integer,Integer>();
		Map<Integer,Integer> idcounts = new TreeMap<Integer,Integer>();
		List<String> usedFiles = new ArrayList<String>();
		
		for (int numberOfIDs = 9; numberOfIDs < 11; numberOfIDs++)
		for (int trainingPerID = 2; trainingPerID < 6; trainingPerID++) {
			Log.d(TAG, "debug: noid:"+numberOfIDs+", tr:"+trainingPerID);
			
			clearIdents();
			idlist.clear();
			idcounts.clear();
			usedFiles.clear();
			helper.shuffleFileArray( FACESfiles );
			
			int count = 0;
			for (int i=0;i<FACESfiles.length;i++) {
				int[] atr = expclass.parseAttributes( FACESfiles[i].getName() );
				int id = atr[expressionclassifier.INDEX_IDENTITY];
				
				boolean add = false;
				count = 0;
				
				// Determine if we need more samples for existing ID or if
				// we can add a new ID
				if (idlist.containsKey(id)) {
					count = idcounts.get(id);
					if (count < trainingPerID) add = true;
				} else if ( idlist.size() < numberOfIDs) add = true;
				
				if (add) {
					Mat face = getFace( FACESfiles[i] );
					if (face == null) continue; //meh
					
					if (!idcounts.containsKey(id)) idcounts.put(id, 0);
					
					int order = 0; //confusing as in FACES name=ID number, in identities ID is just a key for the name
					boolean found = false;
					int maxID = -1;
					for (Entry<Integer,Integer> ID: idlist.entrySet()) {
						if (ID.getValue() > maxID) maxID = ID.getValue();
						if (ID.getKey() == id) {
							order = ID.getValue();
							found = true;
						}
						//Log.d(TAG, "debug: "+ID.getKey()+", "+id+", order:"+order);
					}
					if (!found) order = maxID+1; // new ID
						
					String finalName = identRootDir.getAbsolutePath()+"/"+id+"_"+order+"_"+Integer.toString(generator.nextInt())+".jpg";
										
					Bitmap bitmap = Bitmap.createBitmap(face.width(), face.height(), Bitmap.Config.ARGB_8888);
					Utils.matToBitmap(face, bitmap);
					
					try{
						//Log.d(TAG, "debug: writing file "+finalName);
						FileOutputStream fos = new FileOutputStream( finalName );
						bitmap.compress(Bitmap.CompressFormat.JPEG, 100, fos);
						fos.close();
					} catch (Exception error) { }
					
					idlist.put(id, order);
					idcounts.put(id, idcounts.get(id)+1);
					usedFiles.add(FACESfiles[i].getName());
				}
			}
			
			Log.d(TAG, "debug entries:");
			for (Entry<Integer,Integer> ID: idlist.entrySet()) {
				Log.d(TAG, "debug: "+ID.getKey() + " - "+ID.getValue() + ", count "+idcounts.get(ID.getKey()));
			}
			
			// now retrain
			switch (mIdentifierMethod) {
			case EIGENFACES:
				trainEigenfaceClassifier();
				break;
			case GVLBP:
				trainGVLBPClassifier();
				break;
				default:break;
			}
			
			Log.d(TAG, "debug: Trainded with "+numberOfIDs+" IDs, "+trainingPerID+" faces per ID");
						
			// counters for results
			int processed=0, correct=0, incorrect=0, false_positive=0;
			for (File file: FACESfiles) {
				// skip faces used in training
				boolean bad = false;
				for (String used: usedFiles) if (file.getName().equals(used)) { bad = true; continue; }
				if (bad) continue;
			
				Mat face = getFace( file );
				if (face == null) continue;
				
				int[] result = identifyFace( face );
				
				int recognized = result[expressionclassifier.INDEX_IDENTITY];
				int found_id = -1;
				for (Identity ID: identities) if (ID.getID() == recognized) found_id = Integer.parseInt(ID.getName());
				
				int[] atr = expclass.parseAttributes( file.getName() );
				int true_id = atr[expressionclassifier.INDEX_IDENTITY];
				
				String added = "";
				if (found_id > -1) { // not unknown hit
					if ( true_id == found_id ) {
						correct++;
						added = "positive hit";
					} else {
						false_positive++;
						added = "false positive";
					}
				}
				
				if ( idlist.containsKey(true_id) && (found_id == -1)) {
					incorrect++;
					added = "negative hit";
				}
				processed++;
				
				Log.d(TAG, "debug: processed "+file.getName()+", got ID "+recognized+"("+found_id+") "+added);
			}
			
			String result = "result: error_threshold="+eigen_threshold+", " +eigenfaces_saved+" eigenfaces, ";
			result += Integer.toString(numberOfIDs)+" IDs, "+Integer.toString(trainingPerID);
			result += " faces per ID, correct "+Integer.toString(correct)+", incorrect "+Integer.toString(incorrect);
			result += ", false positives: "+Integer.toString(false_positive);
			result += ", processed:"+Integer.toString(processed);
			result += " = " + Float.toString( (float)correct / (float)(correct+incorrect+false_positive) ) + " recognition rate.";
			Log.d(TAG, result);
			helper.dumpLine(resultFile, result);
		}
		
		clearIdents();
	}
	
	private void deleteFiles( File[] files ) {
		for (File file: files) file.delete();
	}
	
	// deletes every folder and their contents in /pip_idents
	private void clearIdents() {
		File[] files = getJPGList( identRootDir );
				
		if (files != null && files.length > 0) {
			for (File file: files) file.delete();
		}
		updateSampleFiles();
	}
	
	protected File[] getJPGList( File dir ) {
		File[] filelist = dir.listFiles(new FilenameFilter() {
		    public boolean accept(File dir, String name) {
		        return name.toLowerCase().endsWith(".jpg");
		    }
		});
		return filelist;
	}

	// extracts an usable training face from file, meant to be used with 
	// images picked by user as new ID's or FACES-data
	protected Mat getFace(File file) {
		Bitmap bm = null;
		Mat matpic = new Mat();
		try {
			FileInputStream fis = new FileInputStream(file);
			bm = BitmapFactory.decodeFile(file.getAbsolutePath());
			fis.close();
			Utils.bitmapToMat(bm, matpic);
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
				Mat retpic = matpic.submat(faces[fi]).clone();
				Imgproc.cvtColor(retpic, retpic, Imgproc.COLOR_RGB2GRAY);
				Imgproc.equalizeHist(retpic, retpic);
				return retpic;
			}
		}
		
		//helper.savePicture(matpic, null, false, "CRASH_");
		Log.d(TAG, "debug: null face");
		return null;
	}
	
    
}
