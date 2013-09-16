package com.siperia.peopleinphotos;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Scanner;

import org.opencv.android.CameraBridgeViewBase;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfDouble;
import org.opencv.core.MatOfFloat;
import org.opencv.core.MatOfInt;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.ml.CvKNearest;

import delaunay_triangulation.Point_dt;
import android.os.Bundle;
import android.os.Environment;
import android.app.AlertDialog;
import android.app.DialogFragment;
import android.app.Fragment;
import android.app.FragmentTransaction;
import android.content.DialogInterface;
import android.graphics.Bitmap;
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
	
	private static final int		SIM_GRAPH_MATCH = 1, SIM_CONTEXT_MATCH = 2;
	private int 					SIMILARITY_METHOD = SIM_CONTEXT_MATCH;
	private int						currentPhoto = -1;
	private String					picturePath;
	private Map<String,List<Similarity>> similarPhotos = new HashMap<String,List<Similarity>>();
													// list of best matches and their scores
													// First index is for file, second (0-9) are 10 best matches
	private List<Integer>			similarity_weights = new ArrayList<Integer>();
	private CvKNearest				similarity_knn = null;
	
	private ImageView				img;
			
									// How many faces can there be in one photo for graph matching
	private photo					thisPhoto = null;
	private File[]					files;
	
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
			files = faceclass.getJPGList(dir);
			Log.d(TAG, files.length+" photos in "+picturePath);
		}
		
		this_class = this;
		
		loadSimilarities();
		
	}
    
    static GalleryScanner newInstance(FaceDetectionAndProcessing fc, CameraBridgeViewBase mcv) {
        faceclass = fc;
        mCV = mcv;
		
		GalleryScanner f = new GalleryScanner();
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
        img = (ImageView)view.findViewById(R.id.imageView);
        
        Button nextButton = (Button)view.findViewById(R.id.Button_NEXT);
		nextButton.setOnClickListener(new OnClickListener() {
			@Override public void onClick(View v) {
				currentPhoto++;
				if (currentPhoto >= files.length) currentPhoto = 0;
				
				thisPhoto = new photo( files[currentPhoto].getAbsolutePath(), faceclass, true );
				
				bmap = thisPhoto.load();
				img.setImageBitmap(bmap);
												
		    	// Draw the hits on the original picture
				Canvas canvas = new Canvas(bmap);
		        canvas.drawText(thisPhoto.faces.size()+" faces", 10, 40, paint);
		        
		        int i = 0;
		        for (face f: thisPhoto.faces) {
		        	Rect r = f.location;
		        	String tag = "";
		        	
	        		if (faceclass.mIdentifierMethod != FaceDetectionAndProcessing.NO_IDENTIFICATION &&
	        				faceclass.identities.size() > 0) {
		            	if (f.getAttributes()[expressionclassifier.INDEX_IDENTITY] >= 0) tag +=
		            			faceclass.identities.get(f.getAttributes()[expressionclassifier.INDEX_IDENTITY]).getName();
		            	else tag += getResources().getString(R.string.unknown);
		        	}
		        	
		        	switch (f.getAttributes()[expressionclassifier.INDEX_AGE]) {
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
		            
		        	/*paint.setColor(Color.WHITE);
		        	canvas.drawRect((float)r.tl().x-2, (float)r.tl().y-2,
        					(float)r.br().x+2, (float)r.br().y+2, paint);*/
		        	
	            	if (f.getAttributes()[expressionclassifier.INDEX_GENDER] == expressionclassifier.GENDER_MALE) {
	            		canvas.drawRect((float)r.tl().x-2, (float)r.tl().y-2,
			        					(float)r.br().x+2, (float)r.br().y+2, paint);
	            	} else if (f.getAttributes()[expressionclassifier.INDEX_GENDER] == expressionclassifier.GENDER_FEMALE) {
	            		int size = (int)(r.br().x - r.tl().x)/2;
	            		canvas.drawCircle((float)r.tl().x + size, (float)r.tl().y + size, (float)size, paint);
	            	} else {
	            		canvas.drawLine((float)r.tl().x, (float)r.tl().y, (float)r.br().x, (float)r.br().y, paint);
	            	}
	            	
	            	for (Identity ID : faceclass.identities) {
	            		if (f.getAttributes()[expressionclassifier.INDEX_IDENTITY] == ID.getID()) {
	            			canvas.drawText(ID.getName(),(float)r.tl().x, (float)r.tl().y-20, paint);
	            			continue;
	            		}
	            	}	            	
	            	
	            	canvas.drawText(expressionclassifier.expString(f.getAttributes()[expressionclassifier.INDEX_EXPRESSION]),(float)r.tl().x, (float)r.br().y+20, paint);
	            	
	            	i++;
		        }
		        
		        // draw Neighborhood graph
		        paint.setColor(Color.WHITE);
		        for (Pair<Point_dt,Point_dt> edge: thisPhoto.RNG) {
					if (edge != null) if (edge.first != null && edge.second != null) {
						canvas.drawLine((float)edge.first.x(),(float)edge.first.y(),
							(float)edge.second.x(),(float)edge.second.y(),paint);
					}
				}
		        
		        //helper.savePicture(null, bmap, false, "canvas");
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
				calculatePhotoSimilarity(SIMILARITY_METHOD);
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
    				calculatePhotoSimilarity(SIMILARITY_METHOD);
    				
    				dialog.cancel();
    			}
    		});
    		AlertDialog retrainQuestionDialog = builder.create();
    		retrainQuestionDialog.show();
    	}
    	
    	String s="";
    	for (int h=0;h<weights.size();h++) s+= Integer.toString(weights.get(h)) + " ";
    	Log.d(TAG, "debug: similarity weights set to "+s);
    }
    
    protected void showSimilarPhotos() {
    	
    	if ( similarPhotos.get(files[currentPhoto].getAbsolutePath()) != null ) {
	    	showMatch_fragment newFragment = new showMatch_fragment();
	        
			FragmentTransaction ft = getFragmentManager().beginTransaction();
	        Fragment prev = getFragmentManager().findFragmentByTag("showMatch_fragment");
	        if (prev != null) ft.remove(prev);
	        ft.addToBackStack(null);
	        
	        newFragment.setArgs(similarPhotos.get(files[currentPhoto].getAbsolutePath()),picWidth,picHeight);
	        newFragment.show(ft, "showMatch_fragment");
    	} else {
    		Toast.makeText(getActivity(), "No similar photos in file.", Toast.LENGTH_LONG).show();
    		Log.d(TAG, "fcp:"+files[currentPhoto]);
    	}
    	
	}

     
	protected void calculatePhotoSimilarity(int method) {
		similarPhotos.clear();
		
		int ind = 0;
    	for (File base: files) {
			thisPhoto = new photo(base.getAbsolutePath(), faceclass, true);
			List<Similarity> thisPhotoSimilarity = new ArrayList<Similarity>();
			
			if (thisPhoto.faces.size() == 0) {
				// No faces so no point to calculate a similarity score against any other picture
				thisPhotoSimilarity.add( new Similarity( Double.MAX_VALUE, base.getAbsolutePath() ));
				similarPhotos.put(base.getAbsolutePath(), (ArrayList<Similarity>)thisPhotoSimilarity );
				continue;
			}
			
			Log.d(TAG,"findSimilarPhotos, "+thisPhoto.faces.size()+" faces");
			
			double current_mean_face_size = 1;
			if (method == SIM_GRAPH_MATCH) {
				// calculate the mean face size in current photo
		    	for (face f: thisPhoto.faces) {
		    		current_mean_face_size += f.location.width;
		    	}
		    	current_mean_face_size /= thisPhoto.faces.size();
		    	thisPhoto.scaleToMean(1); // convert locations to mean-centered
			}
			
			// compare against every other photo
			for (File file: files) {
				double photoMatchValue = 0;
				
				if ( file.equals(base) ) continue; // skip self hit
				
				// check if this similarity is already calculated
				for (Entry<String, List<Similarity>> e : similarPhotos.entrySet()) {
					for (Similarity s : e.getValue()) {
						if ( file.getAbsolutePath().equals(e.getKey()) || file.getAbsolutePath().equals(s.string) ) {
							if ( base.getAbsolutePath().equals(e.getKey()) || base.getAbsolutePath().equals(s.string) ) {
								// this should be ok as there shouldn't be self pairs in the similarPhotos
								photoMatchValue = s.score;
								thisPhotoSimilarity.add( new Similarity( photoMatchValue, file.getAbsolutePath() ));
							}
						}
						if (photoMatchValue != 0) continue;
					}
					if (photoMatchValue != 0) continue;
				}			
				
				photo compare = new photo( file.getAbsolutePath(), faceclass, true );
				
				if (!compare.found || compare.faces.size() == 0) {
					thisPhotoSimilarity.add( new Similarity( Double.MAX_VALUE, file.getAbsolutePath() ));
				} else if (photoMatchValue == 0) {
					// new photo has been loaded, calculate RNG and Adjasency
					
					switch( SIMILARITY_METHOD ) {
					
					case SIM_GRAPH_MATCH:
						compare.scaleToMean(current_mean_face_size);
						
						int face_count_difference = Math.abs(thisPhoto.faces.size() - compare.faces.size());
						
						double[][] links = calculateBipartitionWeights(thisPhoto, compare);
										
						// Now we should have the full link cost table between current and under comparison photos.
						// To select the best links the Hungarian algorithm is applied.
						
						Hungarian hungarian = new Hungarian( links );
						
						int[] selected = hungarian.execute();
						// This version will return list fitting one row to one column, so selected[0] is the best
						// match for face0 in the photo we are looking matches for. O(n^3) speed
						
						for (int i = 0; i<selected.length; i++) {
							photoMatchValue += links[i][selected[i]];
						}
						// Finally, add the penalty for dissimilar number of faces
						photoMatchValue += similarity_weights.get(4)*face_count_difference;
						break;
					
					case SIM_CONTEXT_MATCH:
						// Calculates graph and group similarities based on "Efficient graph based spatial face context
				    	// representation and matching", which in turn is based on doctoral thesis "Graph matching and its
				    	// application in computer vision and bioinformatics" by Mikhail Zaslavskiy.
							
						// Calculate permutation matrix for property similarity matching
							
						if (compare.adj != null) {
							Log.d(TAG, "debug: sim_context_match");
							
							Mat P = thisPhoto.matchFaces(compare);
							Mat C = Mat.zeros( P.size(), CvType.CV_32FC1);
							
							for (int i = 0;i<thisPhoto.faces.size();i++) {
								for (int j = 0;j<compare.faces.size(); j++) {
									double cost_value = 0;
									
									if (thisPhoto.faces.get(i).attributes[expressionclassifier.INDEX_EXPRESSION] ==
											compare.faces.get(j).attributes[expressionclassifier.INDEX_EXPRESSION])
										cost_value += similarity_weights.get(expressionclassifier.INDEX_EXPRESSION);
									if (thisPhoto.faces.get(i).attributes[expressionclassifier.INDEX_GENDER] ==
											compare.faces.get(j).attributes[expressionclassifier.INDEX_GENDER])
										cost_value += similarity_weights.get(expressionclassifier.INDEX_GENDER);
									if (thisPhoto.faces.get(i).attributes[expressionclassifier.INDEX_AGE] ==
											compare.faces.get(j).attributes[expressionclassifier.INDEX_AGE])
										cost_value += similarity_weights.get(expressionclassifier.INDEX_AGE);
									
									C.put(i, j, cost_value);
								}
							}
							
							Log.d(TAG, "debug: P "+P.dump());
							Log.d(TAG, "debug: C "+C.dump());
							
							//Similarity value
							Mat similarity = new Mat();
							Core.gemm(P, compare.adj, 1, new Mat(), 0, similarity); // sim=PE^y
							Core.gemm(similarity,P,1,new Mat(),0,similarity,Core.GEMM_2_T); // sim=(PE^y)P^t
							Core.subtract(thisPhoto.adj, similarity, similarity); // sim=E^x - P E^y P^t
							
							//Frobenius norm
							Core.gemm(similarity,similarity,1,new Mat(),0,similarity,Core.GEMM_2_T); // E E^t
							Scalar simval =	Core.sumElems(similarity.diag()); // trace(E E^t)
							Log.d(TAG, "simval: "+simval.val[0]);
							photoMatchValue = simval.val[0];
							
							//attribute similarity tr(C^t P) = sum sum C P
							C = C.mul(P);
							Scalar label_sum_s = Core.sumElems(C);
							double label_match = label_sum_s.val[0];
							
							double alpha = 0.5;
							
							photoMatchValue = (1-alpha)*photoMatchValue + alpha*label_match;
						}
					break;
					
					default: break;
					}
					
					thisPhotoSimilarity.add( new Similarity( photoMatchValue, file.getAbsolutePath() ));
				}
			}
			
			Collections.sort( thisPhotoSimilarity );
			//thisPhotoSimilarity = thisPhotoSimilarity.subList(0, Math.min(10,files.length));
			
			similarPhotos.put(base.getAbsolutePath(), thisPhotoSimilarity);
						
			Log.d(TAG, "debug: File "+base.getAbsolutePath()+ " similarities");
			for (Similarity s: thisPhotoSimilarity) {
				Log.d(TAG, s.score + " --- " + s.string);
			}
		}
    	
    	saveSimilarities();
	}

	// weights = distance_alpha, exppression_coeff, gender_coeff, age_coeff, face_count_gamma
	private double[][] calculateBipartitionWeights(photo base, photo comp) {
		if (similarity_weights.size() != 5) {
			Log.e(TAG, "weights.size != 5");
			return new double[0][0];
		}
		
		int smaller = (base.faces.size() <= comp.faces.size()) ? base.faces.size() : comp.faces.size();
		int larger = (base.faces.size() > comp.faces.size()) ? base.faces.size() : comp.faces.size();
		double[][] retval = new double[smaller][larger];
		
		// Every face in photo with fewer faces is matched against faces in the other photo.
		// Depending on weights every link gets a match cost value. Lower value means a better match
		for (int i = 0; i < smaller; i++) {
			for (int j = 0; j < larger; j++) {
				if (base.faces.size() <= comp.faces.size() ) {
					retval[i][j] = calculateLinkCost( base, comp, i, j );
				} else {
					retval[i][j] = calculateLinkCost( base, comp, j, i ); 
				}
			}
		}
						
		return retval;
	}

	// i = index in facesList, j = index in compFacesList
	private double calculateLinkCost(photo base, photo comp, int i, int j) {
		int dx = base.faces.get(i).location.x - comp.faces.get(j).location.x;
		int dy = base.faces.get(i).location.y - comp.faces.get(j).location.y;
		
		double distance = Math.sqrt( Math.pow(dx, 2) + Math.pow(dy, 2)) * similarity_weights.get(0);
		
		double expression = expressionclassifier.exp_dists[base.faces.get(i).getAttributes()[expressionclassifier.INDEX_EXPRESSION]][comp.faces.get(j).getAttributes()[expressionclassifier.INDEX_EXPRESSION]] * similarity_weights.get(expressionclassifier.INDEX_EXPRESSION);
		double gender = Math.abs(base.faces.get(i).getAttributes()[expressionclassifier.INDEX_GENDER] - comp.faces.get(j).getAttributes()[expressionclassifier.INDEX_GENDER]) * similarity_weights.get(expressionclassifier.INDEX_GENDER);
		double age = Math.abs(base.faces.get(i).getAttributes()[expressionclassifier.INDEX_AGE] - comp.faces.get(j).getAttributes()[expressionclassifier.INDEX_AGE])*similarity_weights.get(expressionclassifier.INDEX_AGE);
		
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
					// skip self matches
					if (s.score > 0) bw.write(s.score + delim + s.string + delim);
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
			
			for (File file : files) {
				Log.d(TAG, "file:"+file.getAbsolutePath());
				// Load reasonably down sampled version which will fit into RAM,
				// find faces and calculate graphs and adj. matrix
				photo process = new photo( file.getAbsolutePath(), faceclass, false );
								
				// The absolute path to this photo as ID in file, Number of faces to file
				bw.write( file.getAbsolutePath() + delim + Integer.toString(process.faces.size()) + delim );
				Log.d(TAG, "debug: faces = "+process.faces.size());
				for (int face = 0; face < process.faces.size(); face++) {
					Rect r = process.faces.get(face).location;
					Mat ROI = process.gray.submat(r);

					// Write the location of this face
					bw.write( Integer.toString( r.x ) + delim);
					bw.write( Integer.toString( r.y ) + delim);
					bw.write( Integer.toString( r.width ) + delim);
					bw.write( Integer.toString( r.height ) + delim);
					
		        	int[] result;
		        	Mat hist = new Mat();
		        	
		        	result = faceclass.expclass.identifyExpression(ROI); // resizes to 64x64
		        	faceclass.expclass.nativelib.GaborLBP_Histograms(ROI.clone(), hist, faceclass.LUT, 8, -1, -1);
		        	float[] gaborVals = new MatOfFloat(hist).toArray();
		        	
		        	bw.write( Integer.toString( gaborVals.length ));
		        	for (int h=0;h<gaborVals.length;h++) bw.write( Float.toString(gaborVals[h]) + delim );
		        	
		        	Log.d(TAG, "debug: gabor calc: "+hist.dump());
		        	
		        	// gender, expression and age in order given in expressionclassifier
		        	for (int j=1;j<4;j++) bw.write(Integer.toString(result[j]) + delim );
		        }
				
	        	// neighbor graph size
	        	bw.write( Integer.toString( process.RNG.size() ) + delim );
		        for (Pair<Point_dt,Point_dt> edge: process.RNG) {
		        	bw.write( Double.toString( edge.first.x() ) + delim );
		        	bw.write( Double.toString( edge.first.y() ) + delim );
		        	bw.write( Double.toString( edge.second.x() ) + delim );
		        	bw.write( Double.toString( edge.second.y() ) + delim );
				}
		        
		        // Save the adjacency matrix based on this face order
		        bw.write(Integer.toString(process.RNG.size()));
		        for (int i=0;i<process.adj.size().height;i++)
		        for (int j=0;j<process.adj.size().width;j++) {
		        	double[] in = new double[3];
		        	in = process.adj.get(i, j);
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

	protected void addUsers() {
		int i = 0;
		thisPhoto.getFullSizedSamples();
		
		for (face f : thisPhoto.faces)
	    {
	        addUser_fragment newFragment = new addUser_fragment();
	        
			FragmentTransaction ft = getFragmentManager().beginTransaction();
	        Fragment prev = getFragmentManager().findFragmentByTag("addUser_fragment");
	        if (prev != null) ft.remove(prev);
	        ft.addToBackStack(null);
	        	        
	        //Log.d(TAG, "debug "+ROI+" "+i+" - "+faceclass);
	        newFragment.setArgs(f.getFullPic(), i++, thisPhoto.faces.size(), faceclass);
	        newFragment.show(ft, "addUser_fragment");
	    }
	}
	
	
}
