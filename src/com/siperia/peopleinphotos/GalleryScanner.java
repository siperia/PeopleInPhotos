package com.siperia.peopleinphotos;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.FilenameFilter;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Scanner;

import org.opencv.android.CameraBridgeViewBase;
import org.opencv.android.Utils;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfDouble;
import org.opencv.core.MatOfInt;
import org.opencv.core.MatOfRect;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;
import org.opencv.ml.CvKNearest;

import delaunay_triangulation.BoundingBox;
import delaunay_triangulation.Delaunay_Triangulation;
import delaunay_triangulation.Point_dt;
import delaunay_triangulation.Triangle_dt;

import android.os.Bundle;
import android.os.Environment;
import android.app.AlertDialog;
import android.app.DialogFragment;
import android.app.Fragment;
import android.app.FragmentTransaction;
import android.content.DialogInterface;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.BitmapRegionDecoder;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.util.Log;
import android.util.Pair;

import android.view.LayoutInflater;
import android.view.View;
import android.view.View.OnClickListener;
import android.view.ViewGroup;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.Toast;

public class GalleryScanner extends DialogFragment {
	
	private static final String		TAG = "PeopleInPhotos::GalleryScanner";
	private static final File		sdDir = Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_PICTURES);
	private static final File		identRootDir = new File(sdDir, "PiP_idents");
	private static final String		photoInfoFile = identRootDir.getAbsolutePath() + "/DCIMdata.txt";
	private static final String		matchesFile = identRootDir.getAbsolutePath() + "/matches.txt";
	private static final String		delim = " "; // delimiter used in photoInfoFile
	//private static final String     newline = System.getProperty("line.separator");
	
	private View					view;
	private Paint 					paint;
	private Bitmap					bmap;
	private static int				picWidth;
	private static int				picHeight;
	
	private int						currentPhoto = -1;
	private String					picturePath;
	private File[]					files;			// Files in picturepath folder
	private Map<String,List<Similarity>> similarPhotos = new HashMap<String,List<Similarity>>();
													// list of best matches and their scores
													// First index is for file, second (0-9) are 10 best matches
	private List<Integer>			similarity_weights = new ArrayList<Integer>();
	private CvKNearest				similarity_knn = null;
	
	private ImageView				img;
	private Mat 					gray = new Mat();
	private Mat						rgb	= new Mat();
	
	private MatOfRect				faces =	new MatOfRect();
	private Rect[]					facesList;
	private List<int[]>				facesProperties = new ArrayList<int[]>();
	private Mat						edgeMatrix = null;
		
	private Size 					maxFacesSize = new Size(30,30);
									// How many faces can there be in one photo for graph matching
	private int						facesToCompare;
	private List<String>			compFacesName = new ArrayList<String>();
	private List<Pair<Point_dt,Point_dt>> compFacesList = new ArrayList<Pair<Point_dt, Point_dt>>();
	
	private List<int[]>				compFacesProperties = new ArrayList<int[]>();
	private Mat						compEdgeMatrix = null;
	List<Pair<Point_dt, Point_dt>> 	compRNG = new ArrayList<Pair<Point_dt, Point_dt>>();
	
	private List<Mat>				ROIlist = new ArrayList<Mat>();
		
	List<Pair<Point_dt, Point_dt>> RNG = new ArrayList<Pair<Point_dt, Point_dt>>();
	List<Pair<Point_dt, Point_dt>> notRNG = new ArrayList<Pair<Point_dt, Point_dt>>();
	
    private static FaceDetectionAndProcessing faceclass		= null;
    
    static CameraBridgeViewBase		mCV = null;
    
    private static GalleryScanner	this_class;
    
    public GalleryScanner() {
		picturePath = Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_DCIM).getAbsolutePath();
		picturePath = picturePath + "/Camera";
		
		paint = new Paint();
        paint.setColor(Color.GREEN);
        paint.setStyle(Paint.Style.STROKE);
        paint.setStrokeWidth(2);
        paint.setTextSize(42);
        
        File dir = new File(picturePath);
		if (dir.isDirectory())
		{
			files = dir.listFiles(new FilenameFilter() {
			    public boolean accept(File dir_, String name) {
			        return name.toLowerCase().endsWith(".jpg");
			    }
			});
			Log.d(TAG, files.length+" photos in "+picturePath);
		}
		
		this_class = this;
		
		loadSimilarities();
		
	}
    
    static GalleryScanner newInstance(FaceDetectionAndProcessing fc, CameraBridgeViewBase mcv, int pW, int pH) {
        GalleryScanner f = new GalleryScanner();
        faceclass = fc;
        mCV = mcv;
        picWidth = pW;
		picHeight = pH;
				
        return f;
    }
        
    @Override
    public void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setStyle(DialogFragment.STYLE_NO_TITLE, android.R.style.Theme_Holo_NoActionBar_Fullscreen);
        mCV.disableView();
    }
    
    @Override
    public View onCreateView(LayoutInflater inflater, ViewGroup container,
            Bundle savedInstanceState) {
    	
        view = inflater.inflate(R.layout.activity_gallery_scanner, container);
//        getActivity().getTheme().applyStyle(R.layout.activity_gallery_scanner, true);
        img = (ImageView)view.findViewById(R.id.imageView);
        
        Button nextButton = (Button)view.findViewById(R.id.Button_NEXT);
		nextButton.setOnClickListener(new OnClickListener() {
			@Override public void onClick(View v) {
				currentPhoto++;
				if (currentPhoto >= files.length) currentPhoto = 0;
				
				prepFirst(currentPhoto);
								
		    	// Draw the hits on the original picture
				Canvas canvas = new Canvas(bmap);
		        canvas.drawText(facesList.length+" faces", 10, 40, paint);
		        
		        int i = 0;
		        for (Mat ROI: ROIlist) {
		        	// ROIlist is built in same order than faces are in facesList
		        	Rect r = facesList[i];
		        	String tag = "";
		        			        	
		        	if (faceclass.mIdentifierMethod != FaceDetectionAndProcessing.NO_IDENTIFICATION) {
		            	if (facesProperties.get(i)[expressionclassifier.INDEX_IDENTITY] >= 0) tag +=
		            			faceclass.identities.get(facesProperties.get(i)[expressionclassifier.INDEX_IDENTITY]).getName();
		            	else tag += getResources().getString(R.string.unknown);
		        	}
		        	
		        	switch (facesProperties.get(i)[expressionclassifier.INDEX_AGE]) {
		        	case expressionclassifier.AGE_YOUNG:
		        		paint.setColor(Color.CYAN);
		        		break;
		        	case expressionclassifier.AGE_MIDDLEAGED:
		        		paint.setColor(Color.MAGENTA);
		        		break;
		        	case expressionclassifier.AGE_OLD:
		        		paint.setColor(Color.BLUE);
		        		break;
		        		default:break;
		        	}
		            
	            	if (facesProperties.get(i)[expressionclassifier.INDEX_GENDER] == expressionclassifier.GENDER_MALE) {
	            		canvas.drawRect((float)r.tl().x-2, (float)r.tl().y-2,
			        					(float)r.br().x+2, (float)r.br().y+2, paint);
	            	} else if (facesProperties.get(i)[expressionclassifier.INDEX_GENDER] == expressionclassifier.GENDER_FEMALE) {
	            		int size = (int)(r.br().x - r.tl().x)/2;
	            		canvas.drawCircle((float)r.tl().x + size, (float)r.tl().y + size, (float)size, paint);
	            	} else {
	            		canvas.drawLine((float)r.tl().x, (float)r.tl().y, (float)r.br().x, (float)r.br().y, paint);
	            	}
	            	
	            	paint.setColor(Color.GREEN);
	            	
	            	for (Identity ID : faceclass.identities) {
	            		if (facesProperties.get(i)[expressionclassifier.INDEX_IDENTITY] == ID.getID()) {
	            			canvas.drawText(ID.getName(),(float)r.tl().x, (float)r.tl().y-20, paint);
	            			continue;
	            		}
	            	}
	            	//canvas.drawText(Integer.toString(facesProperties.get(i)[expressionclassifier.INDEX_IDENTITY]),(float)r.tl().x, (float)r.tl().y-20, paint);
	            	
	            	canvas.drawText(expressionclassifier.expString(facesProperties.get(i)[expressionclassifier.INDEX_EXPRESSION]),
	            			(float)r.tl().x, (float)r.br().y+20, paint);
	            	
	            	i++;
		        }
		        		        
		        for (Pair<Point_dt,Point_dt> edge: RNG) {
					canvas.drawLine((float)edge.first.x(),(float)edge.first.y(),
							(float)edge.second.x(),(float)edge.second.y(),paint);
				}
			}
		});
		
		Button allButton = (Button)view.findViewById(R.id.Button_ALL);
		allButton.setOnClickListener(new OnClickListener() {
			@Override public void onClick(View v) {
				addUsers();
			}
		});
		
		Button matchButton = (Button)view.findViewById(R.id.Button_MATCH);
		matchButton.setOnClickListener(new OnClickListener() {
			@Override public void onClick(View v) {
				showSimilarPhotos();
			}			
		});		
		
		Button calcMatchButton = (Button)view.findViewById(R.id.Button_CALC_MATCHES);
		calcMatchButton.setOnClickListener(new OnClickListener() {
			@Override public void onClick(View v) {
				updateImageData();
				calculatePhotoSimilarity();
			}
		});
		
		Button setWeightsButton = (Button)view.findViewById(R.id.Button_SET_WEIGHTS);
		setWeightsButton.setOnClickListener(new OnClickListener() {
			@Override public void onClick(View v) {
				setWeights_fragment newFragment = new setWeights_fragment();
		        
				FragmentTransaction ft = getFragmentManager().beginTransaction();
		        Fragment prev = getFragmentManager().findFragmentByTag("setWeights_fragment");
		        if (prev != null) ft.remove(prev);
		        ft.addToBackStack(null);
		        
		        newFragment.setArgs(this_class, similarity_weights);
		        newFragment.show(ft, "showMatch_fragment");
			}
		});
		
		Button exitButton = (Button)view.findViewById(R.id.Button_EXIT);
		exitButton.setOnClickListener(new OnClickListener() {
			@Override public void onClick(View v) {
				mCV.enableView();
				DialogFragment dialogFragment =
						(DialogFragment)getFragmentManager().findFragmentByTag("GalleryScanner");
				
				if (dialogFragment != null) dialogFragment.dismiss();
			}
		});
		
		nextButton.performClick();
        return view;
    }
    
    public void setWeights( List<Integer> weights ) {
    	boolean changed = false;
    	for (int j=0;j<weights.size();j++) {
    		if (similarity_weights.get(j) != weights.get(j)) changed = true;
    	}
    	
    	similarity_weights = weights;
    	
    	if (changed) {
    		AlertDialog.Builder builder = new AlertDialog.Builder(getActivity());
        	
    		builder.setTitle("Retrain similarity classifier?");
        	builder.setMessage("Current results no longer match the selected weights. Would you like to retrain?");
    		builder.setNegativeButton("No", new DialogInterface.OnClickListener() { 
    			@Override
    			public void onClick(DialogInterface dialog, int id) {
    				dialog.cancel();
    			}
    		});
    		builder.setPositiveButton("Yes", new DialogInterface.OnClickListener() {
    			@Override
    			public void onClick(DialogInterface dialog, int id) {
    				updateImageData();
    				calculatePhotoSimilarity();
    				
    				dialog.cancel();
    			}
    		});
    		AlertDialog retrainQuestionDialog = builder.create();
    		retrainQuestionDialog.show();
    	}
    	
    	String s="";
    	for (int h=0;h<weights.size();h++) s+= Integer.toString(weights.get(h)) + " ";
    	Log.d(TAG, "similarity weights set to "+s);
    }
    
    protected void showSimilarPhotos() {
    	
    	if ( similarPhotos.get(files[currentPhoto].getAbsolutePath()) != null ) {
	    	showMatch_fragment newFragment = new showMatch_fragment();
	        
			FragmentTransaction ft = getFragmentManager().beginTransaction();
	        Fragment prev = getFragmentManager().findFragmentByTag("showMatch_fragment");
	        if (prev != null) ft.remove(prev);
	        ft.addToBackStack(null);
	        
	        newFragment.setArgs(similarPhotos.get(files[currentPhoto].getAbsolutePath()),picWidth,picHeight );
	        newFragment.show(ft, "showMatch_fragment");
    	} else {
    		Toast.makeText(getActivity(), "No similar photos in file.", Toast.LENGTH_LONG).show();
    		Log.d(TAG, "fcp:"+files[currentPhoto]);
    	}
    	
	}

	// loads the "primary" photo and runs facial recognition steps for it. This is the photo
    // shown in galleryview and compared against comp*-variables.
    private void prepFirst(int fileNo) {
    	    	
    	if (loadPic(fileNo, picWidth*2, picHeight*2)) {
			Utils.bitmapToMat(bmap, rgb);
			Imgproc.cvtColor(rgb, gray, Imgproc.COLOR_RGB2GRAY);
			Imgproc.equalizeHist(gray, gray);
									
			faceclass.mCascadeClassifier.detectMultiScale(gray,faces);
			
			List<Rect> flist = faces.toList();
			List<Rect> newList = new ArrayList<Rect>();
			for (Rect l: flist) {
				MatOfRect result = new MatOfRect();
				
				Mat face = rgb.submat(l).clone();
				faceclass.expclass.skinThreshold(face, result, false); // just the pixel count
								
				double skin = (float)result.toList().get(1).x / l.area();
				if ( skin > 0.4 ) { // enough skin to be a face
					newList.add(l);
				}
			}
			MatOfRect mor = new MatOfRect();
			mor.fromList(newList);
			facesList = mor.toArray();  
			
			getFullSizedSamples(fileNo);
				        
	        facesProperties.clear();        
	        int i = 0;
	        
	        for (Mat ROI: ROIlist) {
	        	// ROIlist is built in same order than faces are in facesList
	        	Rect r = facesList[i];
	        		        	
	        	facesProperties.add(faceclass.identifyFace(ROI.clone()));
	        }
	        
	        // calculate the nearest neighbor graph
	        createRNG();
    	} else {
    		Toast.makeText(getActivity(), "Bad image index in loadPic", Toast.LENGTH_LONG).show();
    	}
    }
    
    // scale the compFacesList face location values so the median is the same with the photo we are 
    // currently matching
    protected void scaleToMedian(double scaleTo) {
    	
    	if (compFacesList.size() == 0) return;
    	
    	double comp_median_face_size = 0;
    	//calculate the median value for the current compFacesList
    	if (((compFacesList.size() % 2) == 0) && compFacesList.size() > 0) { //even number of faces
			// median = mean of 2 centermost value
			double div = compFacesList.size() / 2; // = x.5
			int first_ind = (int)Math.floor(div); // x
			int last_ind = (int)Math.ceil(div); // x+1
			comp_median_face_size = (compFacesList.get(first_ind).second.x() + compFacesList.get(last_ind).second.x()) / 2;
			// second.x() == width
		} else {
			int ind = (int)Math.floor(compFacesList.size() / 2);
			if (compFacesList.size() == 1) ind = 0;
			comp_median_face_size = compFacesList.get(ind).second.x();
		}
    	
    	double scaling_factor = scaleTo / comp_median_face_size;
    	//Log.d(TAG, "scaling_factor = "+scaling_factor);
    	
    	double meanX=0,meanY=0;
    	for (int i=0;i<compFacesList.size();i++) {
    		meanX += compFacesList.get(i).first.x();
    		meanY += compFacesList.get(i).first.y();
    	}
    	meanX /= compFacesList.size();
    	meanY /= compFacesList.size();
    	
    	for (int i=0;i<compFacesList.size();i++) {
    		Point_dt location = compFacesList.get(i).first;
    		Point_dt size = compFacesList.get(i).second;
    		
    		// remove mean from locations
    		location.setX( location.x() - meanX );
    		location.setY( location.y() - meanY );
    		
    		// scale the face sizes with scaling factor
    		size.setX( size.x() * scaling_factor );
    		size.setY( size.y() * scaling_factor );
    		
    		// replace the old value
    		compFacesList.set(i, new Pair<Point_dt,Point_dt>(location,size));
    	}
    }
	
     
	protected void calculatePhotoSimilarity() {
		
		similarPhotos.clear();
		
		int ind = 0;
    	for (File base: files) {
			prepFirst(ind++);
			
			List<Similarity> thisPhoto = new ArrayList<Similarity>();
			
			if (facesList.length == 0) {
				// No faces so no point to calculate a similarity score against any other picture
				thisPhoto.add( new Similarity( Double.MAX_VALUE, base.getAbsolutePath() ));
				similarPhotos.put(base.getAbsolutePath(), (ArrayList<Similarity>)thisPhoto );
				continue;
			}
			
			Log.d(TAG,"findSimilarPhotos, "+facesList.length+" faces");
			
			double current_median_face_size = 0;
			// calculate the median face size in current photo
	    	if (((facesList.length % 2) == 0) && facesList.length > 0) { //even number of faces
				// median = mean of 2 centermost value
				double div = facesList.length / 2; // = x.5
				int first_ind = (int)Math.floor(div); // x
				int last_ind = (int)Math.ceil(div); // x+1
				current_median_face_size = (facesList[first_ind].width + facesList[last_ind].width) / 2;
				// faces are always square areas so this will suffice
			} else {
				int index = (int)Math.floor(facesList.length / 2); // 0-based list, 3->1, 5->2, 7->3 etc
				if (facesList.length == 1) index = 0;
				current_median_face_size = facesList[index].width;
			}
			
	    	// compare against every other photo
			for (File file: files) {
				
				if ( file.equals(base) ) {
					thisPhoto.add( new Similarity( 0, file.getAbsolutePath() ));
					continue;
				}				
				
				if (loadImageData(file.getAbsolutePath())) {
					
					scaleToMedian(current_median_face_size);
					
					int face_count_difference = Math.abs(facesList.length - compFacesList.size());
					
					double[][] links = calculateBipartitionWeights();
									
					// Now we should have the full link cost table between current and under comparison photos.
					// To select the best links the Hungarian algorithm is applied.
					
					Hungarian hungarian = new Hungarian( links );
					
					int[] selected = hungarian.execute();
					// This version will return list fitting one row to one column, so selected[0] is the best
					// match for face0 in the photo we are looking matches for. O(n^3) speed
					
					double photoMatchValue = 0;
					//Log.d(TAG, "naamoja: "+facesList.length+", comp naamoja:"+ compFacesList.size()+", selected "+selected.length);
					
					for (int i = 0; i<selected.length; i++) {
						photoMatchValue += links[i][selected[i]];
					}
					// Finally, add the penalty for dissimilar number of faces
					photoMatchValue += similarity_weights.get(4)*face_count_difference;
					
					thisPhoto.add( new Similarity( photoMatchValue, file.getAbsolutePath() ));
				} else {
					thisPhoto.add( new Similarity( Double.MAX_VALUE, file.getAbsolutePath() ));
				}
			}
			
			Collections.sort( thisPhoto );
			thisPhoto = thisPhoto.subList(0, Math.min(10,files.length));
			
			similarPhotos.put(base.getAbsolutePath(), thisPhoto);
			
			/*
			Log.d(TAG, "File "+base.getAbsolutePath()+ " similarities");
			for (Similarity s: thisPhoto) {
				Log.d(TAG, s.score + " --- " + s.string);
			}*/
		}
    	
    	saveSimilarities();
		
		/*
		// Calculates graph and group similarities based on "Efficient graph based spatial face context
    	// representation and matching", which in turn is based on doctoral thesis "Graph matching and its
    	// application in computer vision and bioinformatics" by Mikhail Zaslavskiy.
		Mat this_adj = calcAdjacency(RNG);
		for (File file : files) {
			Log.d(TAG,"file "+file.getName());
			if (loadImageData(file.getAbsolutePath())) {
				// Calculate permutation matrix for property similarity matching
				// Relevant data for this file should be now loaded on "comp"-values
				
				if (compEdgeMatrix != null) {
					Mat P = new Mat();
					P = Mat.zeros(maxFacesSize, CvType.CV_32FC1);
					
					for (int i = 0;i<facesList.length;i++) {
						for (int j = 0;j<compFacesList.size(); j++) {
							int match_value = 0;
							if (faceProperties[INDEX_IDENTITY] == compFacesProperties[j][INDEX_IDENTITY]) match_value=1;							
							//if (faceProperties[INDEX_GENDER] == compFacesProperties[j][INDEX_GENDER]) match_value++;
							//if (faceProperties[INDEX_AGE] == compFacesProperties[j][INDEX_AGE]) match_value++;
							
							P.put(i, j, (double)match_value);
						}
					}
					
					Log.d(TAG, "Pnorm:"+P.dump());
					
					//Similarity value
					Mat similarity = new Mat();
					Core.gemm(P, compEdgeMatrix, 1, new Mat(), 0, similarity); // sim=PE^y
					Core.gemm(similarity,P,1,new Mat(),0,similarity,Core.GEMM_2_T); // sim=(PE^y)P^t
					Core.subtract(this_adj, similarity, similarity); // sim=E^x - P E^y P^t
					
					//Frobenius norm
					Core.gemm(similarity,similarity,1,new Mat(),0,similarity,Core.GEMM_2_T); // E E^t
					Scalar simval =	Core.sumElems(similarity.diag());
					Log.d(TAG, "simval: "+simval.val[0]);
					
					compEdgeMatrix = null;
				}
				
			}
		}*/
	}

	// weights = distance_alpha, exppression_coeff, gender_coeff, age_coeff, face_count_gamma
	private double[][] calculateBipartitionWeights() {
		if (similarity_weights.size() != 5) {
			Log.e(TAG, "weights.size != 5");
			return new double[0][0];
		}
		
		int smaller = (facesList.length <= compFacesList.size()) ? facesList.length : compFacesList.size();
		int larger = (facesList.length > compFacesList.size()) ? facesList.length : compFacesList.size();
		double[][] retval = new double[smaller][larger];
		
		// Every face in photo with fewer faces is matched against faces in the other photo.
		// Depending on weights every link gets a match cost value. Lower value means a better match
		for (int i = 0; i < smaller; i++) {
			for (int j = 0; j < larger; j++) {
				if (facesList.length <= compFacesList.size() ) {
					retval[i][j] = calculateLinkCost( i,j );
				} else {
					retval[i][j] = calculateLinkCost( j,i ); 
				}
			}
		}
						
		return retval;
	}

	// i = index in facesList, j = index in compFacesList
	private double calculateLinkCost(int i, int j) {
		Point_dt p1 = new Point_dt( facesList[i].x, facesList[i].y );
		double distance = p1.distance( compFacesList.get(j).first ) * similarity_weights.get(0);
		
		double expression = expressionclassifier.exp_dists[facesProperties.get(i)[expressionclassifier.INDEX_EXPRESSION]][compFacesProperties.get(j)[expressionclassifier.INDEX_EXPRESSION]] * similarity_weights.get(1);
		double gender = Math.abs(facesProperties.get(i)[expressionclassifier.INDEX_GENDER] - compFacesProperties.get(j)[expressionclassifier.INDEX_GENDER]) * similarity_weights.get(2);
		double age = Math.abs(facesProperties.get(i)[expressionclassifier.INDEX_AGE] - compFacesProperties.get(j)[expressionclassifier.INDEX_AGE])*similarity_weights.get(3);
		
		return distance + expression + gender + age;
	}
	
	private void trainKNN() {
		Mat samples = new Mat();
		Mat classes = new Mat();
		
		int index = 0;
		for (File file: files) {
			List<Similarity> sims = similarPhotos.get( file.getAbsolutePath() );
			if (sims != null) {
				for (Similarity s: sims) {
					// these are stored in ascending order so they can be inserted as is
					
					int ind = -1;
					for (int i=0;i<files.length;i++) if (files[i].equals(s.string)) { ind = i; continue; } 
					
					if (ind >= 0) {
						samples.push_back(new MatOfDouble(s.score));
						classes.push_back(new MatOfInt(ind));
					}
				}
			}
			
			
			index++;
		}
		
		
	}

	// save the best matches for each photo
	private void saveSimilarities() {
		try {
			File file = new File(matchesFile);
			if (!file.exists()) file.createNewFile();
			
			FileWriter fw = new FileWriter(file, false);
			BufferedWriter bw = new BufferedWriter(fw);
			
			bw.write(Integer.toString(similarity_weights.size())+delim);
			for (int i: similarity_weights) {
				bw.write(Integer.toString(i) + delim);
			}
			bw.newLine();

			for (Entry<String,List<Similarity>> sim: similarPhotos.entrySet()) {
				bw.write(sim.getKey() + delim);
				
				for (Similarity s: sim.getValue()) {
					bw.write(s.score + delim + s.string + delim);
				}
				bw.newLine();
			}
			bw.close();
			fw.close();
		} catch (Exception e) {
			e.printStackTrace();
		}
	}
		
	private void loadSimilarities() {
		
		similarPhotos.clear();
		String inKey;
		Similarity inPair;
	
		try {
			FileReader fileReader = new FileReader(new File(matchesFile));
        	Scanner scan = new Scanner( fileReader );
        	
        	int i = scan.nextInt();
        	for (int k=0;k<i;k++)
        		similarity_weights.add(scan.nextInt());
        		
        	while (scan.hasNext()) {
        		inKey = scan.next();
        		
        		Scanner innerScan = new Scanner( scan.nextLine() );
        		List<Similarity> inList = new ArrayList<Similarity>();
        		
        		while (innerScan.hasNext()) {
        			inPair = new Similarity( innerScan.nextDouble(), innerScan.next() );
        			inList.add(inPair);
        		}
        		
        		similarPhotos.put( inKey, inList );
        	}
        	fileReader.close();
		} catch (Exception e) {
			e.printStackTrace();
		}
	}
	
	// Iterate through the entire folder and save the results in photoInfoFile-file
	private void updateImageData() {
		try {
			File photoFile = new File(photoInfoFile);
			if (!photoFile.exists()) photoFile.createNewFile();
						
			FileWriter fw = new FileWriter(photoFile,false);
			BufferedWriter bw = new BufferedWriter(fw);
			
			Mat this_rgb = new Mat(), this_gray = new Mat();
			MatOfRect faces = new MatOfRect();
			Rect[] thisFacesList;
			
			for (File file : files) {
				// The absolute path to this photo as ID in file
				
				bw.write( file.getAbsolutePath() + delim );
				
				// Load reasonably down sampled version which will fit into RAM
				Bitmap bmap = decodeSampledBitmapFromFile(file,picWidth,picHeight);
				
				Utils.bitmapToMat(bmap, this_rgb);
				Imgproc.cvtColor(this_rgb, this_gray, Imgproc.COLOR_RGB2GRAY);
				Imgproc.equalizeHist(this_gray, this_gray);
				
				int finalWidth = this_gray.width();
				int finalHeight = this_gray.height();
				
				faceclass.mCascadeClassifier.detectMultiScale(this_gray, faces);
				thisFacesList = faces.toArray();
				
				// Number of faces to file
				bw.write( Integer.toString(thisFacesList.length) + delim );
				for (int face = 0; face < thisFacesList.length; face++) {
					Rect r = thisFacesList[face];
					Mat ROI = this_gray.submat(r);

					// Write the location of this face
					bw.write( Integer.toString( r.x ) + delim);
					bw.write( Integer.toString( r.y ) + delim);
					bw.write( Integer.toString( r.width ) + delim);
					bw.write( Integer.toString( r.height ) + delim);
					
		        	int[] result;
		        	if (faceclass.mIdentifierMethod != FaceDetectionAndProcessing.NO_IDENTIFICATION) {
		        		faceclass.mIdentifierMethod = FaceDetectionAndProcessing.EIGENFACES;
		        		result = faceclass.identifyFace(ROI);
		        		faceclass.mIdentifierMethod = FaceDetectionAndProcessing.NO_IDENTIFICATION;
		        	} else {
		        		result = faceclass.identifyFace(ROI);
		        	}
		        	
		        	if (result[0] >= 0) bw.write( faceclass.identities.get(result[0]).getName() + delim );
	            	else bw.write( "unknown" + delim );

		        	// gender, expression and age in order given in expressionclassifier
		        	for (int j=1;j<4;j++) bw.write(Integer.toString(result[j]) + delim );
		        }
				
				createRNG();
	        	// neighbor graph size
	        	bw.write( Integer.toString( RNG.size() ) + delim );
	        	// and the normalized edges
		        for (Pair<Point_dt,Point_dt> edge: RNG) {
		        	bw.write( Double.toString( (double)edge.first.x() / finalWidth ) + delim );
		        	bw.write( Double.toString( (double)edge.first.y() / finalHeight ) + delim );
		        	bw.write( Double.toString( (double)edge.second.x() / finalWidth ) + delim );
		        	bw.write( Double.toString( (double)edge.second.y() / finalHeight ) + delim );
				}
		        
		        // Calculate the adjacency matrix based on this face order
		        //Log.d(TAG, "rng size:"+RNG.size());
		        Mat adj = calcAdjacency(RNG);
		        
		        bw.write(Integer.toString(RNG.size()));
		        for (int i=0;i<RNG.size();i++)
		        for (int j=0;j<RNG.size();j++) {
		        	double[] in = new double[3];
		        	in = adj.get(i, j);
		        	bw.write( Integer.toString((int)in[0])+delim);
		        }
		        
				bw.newLine();
			}
			bw.flush(); fw.flush();
			bw.close();	fw.close();
		} catch (Exception e) {
			e.printStackTrace();
		}
	}
	
	private boolean loadImageData(String this_file) {
		// Reads previously classified image data from photoInfoFile-file.
		// The structure per line is
		// absolutePath(String) numberOfFaces(int) {
		//	(normalized centered face coords) tl.x(double) tl.y(double) faceName(string) faceGender(int) faceExpression(int) faceAge(int)
		// } RNGedgesCount(int) {
		//	vertex1[i].x vertex1[i].y vertex2[i].x vertex2[i].y
		// }
		
		boolean found = false;
		try
        {
            FileReader fileReader = new FileReader(new File(photoInfoFile));
            Scanner scan = new Scanner( fileReader );
            
            while (!found && scan.hasNext()) {
            	String fname = scan.next();
            	if (fname.equals(this_file)) found = true; else scan.nextLine();
            }
            if (found) {
            	compFacesList.clear();
            	compFacesName.clear();
            	compRNG.clear();
            	
	            facesToCompare = scan.nextInt();
	            compFacesProperties.clear();
	            
	            int[] toadd = new int[4];
	        	for (int face=0;face < facesToCompare;face++)
	        	{
	        		int x=scan.nextInt();
	        		int y=scan.nextInt();
	        		int w=scan.nextInt();
	        		int h=scan.nextInt();
	        		
	        		compFacesList.add( new Pair<Point_dt,Point_dt>(new Point_dt(x,y), new Point_dt(w,h) ) );
	        		
	        		compFacesName.add( scan.next() );
	        		
	        		toadd[expressionclassifier.INDEX_IDENTITY] = 0; //placeholder
	        		toadd[expressionclassifier.INDEX_EXPRESSION] = scan.nextInt();
	        		toadd[expressionclassifier.INDEX_GENDER] = scan.nextInt();
	        		toadd[expressionclassifier.INDEX_AGE] = scan.nextInt();
	        		
	        		compFacesProperties.add(toadd);
	        	}
	        	
	        	int graphEdges = scan.nextInt();
	        	compRNG.clear();
	        	for (int edge=0;edge<graphEdges;edge++) {
	        		compRNG.add(new Pair<Point_dt, Point_dt>(new Point_dt(scan.nextDouble(),scan.nextDouble()),new Point_dt(scan.nextDouble(),scan.nextDouble())));
	        	}
	        	
	        	compEdgeMatrix = Mat.zeros(maxFacesSize, CvType.CV_32FC1);
	        	for (int i=0;i<graphEdges;i++) {
	        		for (int j=0;j<graphEdges;j++) {
	        			int adjval = scan.nextInt();
	        			compEdgeMatrix.put(i, j, adjval);
	        		}	        		
	        	}
	    		//Log.d(TAG, compEdgeMatrix.dump());
	        	
	        	if (facesToCompare == 0) return false;
	    		
            }
            scan.close();
        }
        catch ( Exception e )
        {
            e.printStackTrace();
        }

		return found;
	}
	
	private void getFullSizedSamples(int fileNo) {
		try {
			FileInputStream is = new FileInputStream(files[fileNo]);
			ROIlist.clear();
			
			BitmapRegionDecoder brd = BitmapRegionDecoder.newInstance(is, true);
			BitmapFactory.Options opts = new BitmapFactory.Options();
			opts.outHeight = 200;
			opts.outWidth = 200;
			
			final int width = brd.getWidth();
			final int height = brd.getHeight();
			for (Rect ROI: facesList) {
				Mat matpic = new Mat();
				android.graphics.Rect androidRect = new android.graphics.Rect(
						(int)ROI.tl().x*(width/rgb.width()), (int)ROI.tl().y*(height/rgb.height()),
						(int)ROI.br().x*(width/rgb.width()), (int)ROI.br().y*(height/rgb.height()));
								
				Bitmap bit_in = brd.decodeRegion(androidRect, opts);
				Utils.bitmapToMat(bit_in, matpic);
				
				//helper.savePicture(matpic, false, "full");
				ROIlist.add(matpic);
			}
			is.close();
		} catch ( Exception e ) {
			e.printStackTrace();
		}		
	}
	
	
	
	// Copy'n'paste from http://developer.android.com/training/displaying-bitmaps/load-bitmap.html
	public static int calculateInSampleSize(
		BitmapFactory.Options options, int reqWidth, int reqHeight) {
	    // Raw height and width of image
	    final int height = options.outHeight;
	    final int width = options.outWidth;
	    int inSampleSize = 1;
	    if (height > reqHeight || width > reqWidth) {
	        // Calculate ratios of height and width to requested height and width
	        final int heightRatio = Math.round((float) height / (float) reqHeight);
	        final int widthRatio = Math.round((float) width / (float) reqWidth);
	        // Choose the smallest ratio as inSampleSize value, this will guarantee
	        // a final image with both dimensions larger than or equal to the
	        // requested height and width.
	        inSampleSize = heightRatio < widthRatio ? heightRatio : widthRatio;
	    }
    	return inSampleSize;
	}
    
    public static Bitmap decodeSampledBitmapFromFile(File file, int reqWidth, int reqHeight)
    {
        // First decode with inJustDecodeBounds=true to check dimensions
        final BitmapFactory.Options options1 = new BitmapFactory.Options();
        options1.inJustDecodeBounds = true;
        BitmapFactory.decodeFile(file.getAbsolutePath(),options1);

        // Calculate inSampleSize
        options1.inSampleSize = calculateInSampleSize(options1, reqWidth, reqHeight);
        // Decode bitmap with inSampleSize set
        options1.inJustDecodeBounds = false;
        // Needed for canvas editing
        options1.inMutable = true;        
        
        return BitmapFactory.decodeFile(file.getAbsolutePath(),options1);
    }


	protected boolean loadPic(int which, int width, int height) {
		if (which >= files.length || which < 0) return false;
		
		if (files.length > 0) {
			bmap = decodeSampledBitmapFromFile(files[which], width, height);
			img.setImageBitmap(bmap);
		} else {
			bmap = BitmapFactory.decodeResource(getResources(), R.raw.lena);
			img.setImageBitmap(bmap);
			Log.d(TAG,"Opening Lena.jpg");
		}
		return true;
	}

	protected void addUsers() {
		int i = 0;
		for (Mat ROI : ROIlist)
	    {
	        addUser_fragment newFragment = new addUser_fragment();
	        
			FragmentTransaction ft = getFragmentManager().beginTransaction();
	        Fragment prev = getFragmentManager().findFragmentByTag("addUser_fragment");
	        if (prev != null) ft.remove(prev);
	        ft.addToBackStack(null);
	        
	        /*if (mListener != null) mListener.sendMat(ROI); 
	        setMatListener(new GalleryScanner.SendMat() {
				@Override public void sendMat(Mat payload) {
					newFragment.useFaceMat(facepic)
				}
			})*/

	        newFragment.setArgs(ROI.clone(), i++, ROIlist.size(), faceclass);
	        newFragment.show(ft, "addUser_fragment");
	    }
	}
	
	private Point rectCenter( Rect A ) {
		return new Point(A.x + (A.width / 2), A.y + (A.height / 2));
	}
	
	private Point_dt rectCenter_dt( Rect A ) {
		return new Point_dt(A.x + (A.width / 2), A.y + (A.height / 2));
	}
	
	private void createRNG() {
		Delaunay_Triangulation DTri = new Delaunay_Triangulation();
		RNG.clear();
		notRNG.clear();
		
		if (facesList.length > 2) {
			for (int i=0; i < facesList.length; i++) {
				Point face = rectCenter( facesList[i] );
				Point_dt dt = new Point_dt(face.x, face.y);
				
				DTri.insertPoint(dt);
			}
			
			Iterator<Triangle_dt> T_iter = DTri.trianglesIterator();
			
			while (T_iter.hasNext()) {
				Triangle_dt triangle = T_iter.next();
				
				// reject triangles with vertices outside the frame as these
				// are the "support" triangles forming the convex outer rim.
				if (triangle.p1() != null && triangle.p2() != null && triangle.p3() != null) {
					BoundingBox bb = triangle.getBoundingBox();
					if (bb.minX() >= 0 && bb.minY() >= 0 && bb.maxX() < rgb.width() && bb.maxY() < rgb.height()) {
						Point_dt v1,v2;
						
						double ed1=processEdge( triangle, triangle.next_12(), 1 ); // process edge 1-2 etc
						double ed2=processEdge( triangle, triangle.next_23(), 2 );
						double ed3=processEdge( triangle, triangle.next_31(), 3 );
						
						if (ed1 > ed2 && ed1 > ed3) { v1=triangle.p1(); v2=triangle.p2(); }
						else if (ed2 > ed1 && ed2 > ed3) {v1=triangle.p2(); v2=triangle.p3(); }
						else {v1=triangle.p3(); v2=triangle.p1(); }
						notRNG.add( new Pair<Point_dt,Point_dt>(v1,v2));
					}
				}
			}
		} else if (facesList.length == 2) {
			Point face1 = rectCenter( facesList[0] ); Point_dt f1 = new Point_dt(face1.x,face1.y);
			Point face2 = rectCenter( facesList[1] ); Point_dt f2 = new Point_dt(face2.x,face2.y);
			RNG.add(new Pair<Point_dt,Point_dt>( f1, f2 ));
		}
		
		// now we have RNG with hits from each triangle. Next we need to remove the longest edges
		// from each triangle and those which where rejected in other neighboring triangles.
		for (Pair<Point_dt, Point_dt> bad_case : notRNG) {
			while (RNG.remove(bad_case));
			while (RNG.remove(new Pair<Point_dt,Point_dt>(bad_case.second, bad_case.first)));
			
			// Edge vertices are not guaranteed to be in any orientation
			// There can also be several occurrences of the same edge.
		}
		
		Log.d(TAG, "size of trian:"+DTri.trianglesSize()+", RNG:"+RNG.size()+", size of bad:"+notRNG.size());
	}
	
	private Mat calcAdjacency( List<Pair<Point_dt,Point_dt>> edges ) {		
		Mat adj = new Mat(maxFacesSize, CvType.CV_32FC1);
		adj = Mat.zeros(maxFacesSize, CvType.CV_32FC1);
		
		for (int edge = 0; edge<edges.size(); edge++) {
			int face1_ind=255, face2_ind=255;
			
			for (int f1=0;f1<facesList.length;f1++) {
				if ( rectCenter_dt( facesList[f1] ).equals(edges.get(edge).first) ) {
					face1_ind = f1;
				}
				if ( rectCenter_dt( facesList[f1] ).equals(edges.get(edge).second) ) {
					face2_ind = f1;
				}
			}
			
			if ((face1_ind != 255) && (face2_ind != 255)) {
				adj.put(face1_ind, face2_ind, 1); // these 2 edges are adjacent
				adj.put(face2_ind, face1_ind, 1); //and symmetry
			}
        }
		return adj;
	}
	
	private double processEdge( Triangle_dt triangle, Triangle_dt neighbor, int edge ) {
		if (triangle == null) return 0;
		
		Point_dt common1 = null, common2 = null, tri_3rd = null, other = null;
		double dist_self,dist_other_1,dist_other_2,dist_neigh_1, dist_neigh_2, max=0;
		
		// pick the vertices used in this edge
		switch (edge) {
		case 1:
			common1 = triangle.p1(); common2 = triangle.p2(); tri_3rd = triangle.p3();
			break;
		case 2:
			common1 = triangle.p2(); common2 = triangle.p3(); tri_3rd = triangle.p1();
			break;
		case 3:
			common1 = triangle.p3(); common2 = triangle.p1(); tri_3rd = triangle.p2();
			break;
		default:break;
		}
		
		dist_self = common1.distance(common2);
		
		dist_other_1 = common1.distance(tri_3rd);
		dist_other_2 = common2.distance(tri_3rd);
		if (dist_other_2 > dist_other_1) dist_other_1 = dist_other_2;
		
		// and if neighbor is a valid triangle, get the distances to its 3rd point.
		// This step can still "see" the support triangles so skip those
		max = dist_other_1;
		if (neighbor != null && neighbor.p1() != null && neighbor.p2() != null && neighbor.p3() != null) {
			BoundingBox bb = neighbor.getBoundingBox();
			if (bb.minX() >= 0 && bb.minY() >= 0 && bb.maxX() < rgb.width() && bb.maxY() < rgb.height()) {
				if (triangle.isCorner(neighbor.p1()) && triangle.isCorner(neighbor.p2())) other = neighbor.p3();
				if (triangle.isCorner(neighbor.p2()) && triangle.isCorner(neighbor.p3())) other = neighbor.p1();
				if (triangle.isCorner(neighbor.p3()) && triangle.isCorner(neighbor.p1())) other = neighbor.p2();
				
				dist_neigh_1 = common1.distance(other);
				dist_neigh_2 = common2.distance(other);
				dist_neigh_1 = (dist_neigh_1 > dist_neigh_2) ? dist_neigh_1 : dist_neigh_2;
				if (dist_neigh_2 > dist_neigh_1) dist_neigh_1 = dist_neigh_2; 
				if (dist_neigh_1 > max) max = dist_neigh_1;
			} 
		}
		
		if (dist_self < max) RNG.add(new Pair<Point_dt,Point_dt>(common1,common2));
		else notRNG.add(new Pair<Point_dt,Point_dt>(common1,common2));
		
		return dist_self;
	}
}
