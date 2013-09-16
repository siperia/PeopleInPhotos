package com.siperia.peopleinphotos;

//import android.os.Bundle;
//import android.R;
import android.app.Activity;
import android.app.AlertDialog;
import android.app.Dialog;
import android.app.Fragment;
import android.app.FragmentTransaction;
import android.app.AlertDialog.Builder;
import android.view.Menu;

import java.io.File;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Iterator;
import java.util.List;
import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.CameraBridgeViewBase.CvCameraViewFrame;
import org.opencv.android.LoaderCallbackInterface;
import org.opencv.android.OpenCVLoader;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfDouble;
import org.opencv.core.MatOfFloat;
import org.opencv.core.MatOfInt;
import org.opencv.core.MatOfRect;
import org.opencv.core.Point;
import org.opencv.core.Size;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.imgproc.Imgproc;
import org.opencv.android.CameraBridgeViewBase;
import org.opencv.android.CameraBridgeViewBase.CvCameraViewListener2;
import android.content.Context;
import android.content.DialogInterface;
import android.graphics.Color;
import android.hardware.Camera;
import android.os.Bundle;
import android.os.Environment;
import android.util.Log;
import android.view.MenuItem;
import android.view.MotionEvent;
import android.view.View;
import android.view.View.OnClickListener;
import android.view.WindowManager;
import android.widget.Button;
import android.widget.SeekBar;
import android.widget.TextView;
import android.widget.Toast;
import android.widget.SeekBar.OnSeekBarChangeListener;

import com.siperia.peopleinphotos.expressionclassifier;

public class MainActivity extends Activity implements CvCameraViewListener2  {

    private static final String    TAG					= "PeopleInPhotos::MainActivity";
    private static final File 	   sdDir = Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_PICTURES);
            	
    private static Scalar 		   FACE_RECT_COLOR		= new Scalar(0,255,0,255);
    private static Scalar		   NOTICE_TEXT_COLOR	= new Scalar(255,30,0,255);
    private static Scalar		   BLACK				= new Scalar(0,0,0,255);
    
    public static final int		   NO_PREPROCESSING		= 0;
    public static final int		   FULL_PIC_HISTOGRAMEQ = 1;
    public static final int		   SKINCOLOUR_THRESHOLD	= 2; // 4,8,etc
    private boolean[]			   selectedPreprocessor = new boolean[3];
    							       
    private FaceDetectionAndProcessing faceclass		= null;
    
    private MenuItem	           mItemSystem;
    private MenuItem	           mItemTriggering;
    private MenuItem	           mItemRegocnition;
    private MenuItem			   mIdentification;
    private MenuItem	           mItemPreprocessing;
    private MenuItem	           mItemTesting;

    private Mat                    mRgba;
    private Mat                    mGray;
    private Mat					   facepic				= null;
    private Rect[]				   facesArray 			= null;

    private Context				   mainActivityContext = this;
    
    private boolean[]		       selectedTriggerItems = new boolean[6];
    private static final int 	   TOUCH_TRIGGER = 0, SMILE_TRIGGER=1,GROUP_TRIGGER=2,
    							   SPESIFIC_TRIGGER=3,NEWID_TRIGGER=4,MAX_RATE_TRIGGER=5;
    
    private List<Point> 		   desireable_previews = new ArrayList<Point>();
    private Point 				   photoSize = new Point(640,480);
        
    private int					   maxPhotoRate=30; //default

    private CameraBridgeViewBase   mOpenCvCameraView;
    
    
    private BaseLoaderCallback  mLoaderCallback = new BaseLoaderCallback(this) {
        @Override
        public void onManagerConnected(int status) {
            switch (status) {
                case LoaderCallbackInterface.SUCCESS:
                {                	
                    Log.i(TAG, "OpenCV loaded successfully");
                    
                    String[] libraries = {"detection_based_tracker","expression"};

                    for (String lib: libraries) {
                    	try{
                    		System.loadLibrary(lib);
	                    } catch(UnsatisfiedLinkError e) {
	                        Log.e(TAG, "Couldn't load "+lib+" library!");
	                        e.printStackTrace();
	                    }
                    }
                    // load cascade file from application resources (Viola-Jones)
                    faceclass = new FaceDetectionAndProcessing(mainActivityContext);
                    faceclass.selectFaceRecognicer( FaceDetectionAndProcessing.LBPBASED );
                    
                    // Populate the list of available video preview resolutions
                	Camera cam = Camera.open();
                	Camera.Parameters par = cam.getParameters(); 
        			List<Camera.Size> sizes = par.getSupportedPreviewSizes();
        			cam.release();
        			cam=null;
                	Iterator<Camera.Size> itr = sizes.iterator();
                	while(itr.hasNext())                    	
                	{
                		Camera.Size s = itr.next();
                		desireable_previews.add(new Point(s.width, s.height));
                	}                	
                	
                	// Screentouch init
                	final View touchView = findViewById(R.id.main_activity_surface_view);
                    touchView.setOnTouchListener(new View.OnTouchListener() {
                        @Override
                        public boolean onTouch(View v, MotionEvent event) {
                        	if (event.getAction() == MotionEvent.ACTION_DOWN)
                        	{
                            	if (selectedTriggerItems[TOUCH_TRIGGER])
                            	{
                            		helper.savePicture(mRgba, null, true, "TouchTriggered");
                            	} else {
                            		// Grab teaching samples from people currently in frame
                            		for (int i = facesArray.length-1; i > -1; i--)
                            	    {
                            			Mat ROI = mRgba.submat(new Rect((int)facesArray[i].tl().x,
                            	        						(int)facesArray[i].tl().y,
                            	        						facesArray[i].width,
                            	    						    facesArray[i].height));
                            			
                            			FragmentTransaction ft = getFragmentManager().beginTransaction();
                            	        Fragment prev = getFragmentManager().findFragmentByTag("addUser_fragment");
                            	        if (prev != null) {
                            	            ft.remove(prev);
                            	        }
                            	        
                            	        ft.addToBackStack(null);
                            	        addUser_fragment newFragment = new addUser_fragment();
                            	        newFragment.setArgs(ROI.clone(), i, facesArray.length-1, faceclass);
                            	        
                            	        newFragment.show(ft, "addUser_fragment");
                            	    }
                            	}                                
                        	}
                        	return true;
                        }
                    });
                                        
                    mOpenCvCameraView.enableView();
                } break;
                default:
                {
                    super.onManagerConnected(status);
                } break;
            }
        }
    };
    
    public MainActivity() {
        Log.i(TAG, "Instantiated new " + this.getClass());
        
    }

	/** Called when the activity is first created. */
    @Override
    public void onCreate(Bundle savedInstanceState) {
        Log.i(TAG, "called onCreate");
        super.onCreate(savedInstanceState);
        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);

        setContentView(R.layout.activity_main);

        mOpenCvCameraView = (CameraBridgeViewBase) findViewById(R.id.main_activity_surface_view);
        mOpenCvCameraView.setCvCameraViewListener(this);
        
        selectedPreprocessor[FULL_PIC_HISTOGRAMEQ] = true;
    }
    
    @Override
    public void onPause()
    {
    	super.onPause();
        if (mOpenCvCameraView != null)
            mOpenCvCameraView.disableView();
    }

    @Override
    public void onResume()
    {
        super.onResume();
        OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION_2_4_3, this, mLoaderCallback);
    }

    public void onDestroy() {
        super.onDestroy();
        mOpenCvCameraView.disableView();
    }

    public void onCameraViewStarted(int width, int height) {
        mGray = new Mat();
        mRgba = new Mat();
    }

    public void onCameraViewStopped() {
        mGray.release();
        mRgba.release();
    }

    // The method where the actual processing is done
    public Mat onCameraFrame(CvCameraViewFrame inputFrame) {
        mRgba = inputFrame.rgba();
        mGray = inputFrame.gray().clone(); // clone to prevent flickering
        int x_offset = 0, y_offset = 0;
        
        if (facepic == null) facepic = new Mat(faceclass.facesize, CvType.CV_8U);
                
        // Thresholding to find areas which are skincoloured        
        if (selectedPreprocessor[SKINCOLOUR_THRESHOLD]) {
        	MatOfRect clip = new MatOfRect();
        	faceclass.skinThreshold(mRgba, clip, true);
        	Rect[] clipplane = clip.toArray();
        	
        	if ((clipplane[0].width < mRgba.width()) || (clipplane[0].height < mRgba.height())) {
        		Core.rectangle(mRgba, new Point(clipplane[0].tl().x, clipplane[0].tl().y),
            			new Point(clipplane[0].br().x, clipplane[0].br().y), FACE_RECT_COLOR, 1);
        	
	        	mGray = mGray.submat(clipplane[0]);
	        	// every coordinate in mRgba is now relative to mGray+offsets 
	        	x_offset = (int)clipplane[0].tl().x;
	        	y_offset = (int)clipplane[0].tl().y;
        	}
        }
        
        if (selectedPreprocessor[FULL_PIC_HISTOGRAMEQ]) Imgproc.equalizeHist(mGray,mGray);

        // Lets check if the camera is taking frames upside down and flip the frame if so
        /*int rotation = getWindowManager().getDefaultDisplay().getRotation();
        if(rotation == Surface.ROTATION_270) {
			Core.flip(mRgba, mRgba, -1);
			Core.flip(mGray, mGray, -1);
		}*/
        
        MatOfRect faces = new MatOfRect();
        if (faceclass.mNativeDetector != null) faceclass.mNativeDetector.detect(mGray, faces);
        facesArray = faces.toArray();
    	// and draw the hits on the picture
        
        final int fontFace = Core.FONT_HERSHEY_PLAIN;
    	final double fontScale = 2;
    	final int thickness = 3;
    	final int[] baseLine = {0};

        if (facesArray.length == 0) { // no faces text
        	Core.putText(mRgba, getResources().getString(R.string.no_face_found), new Point(15, mRgba.height()-10), Core.FONT_HERSHEY_DUPLEX, 1.5, NOTICE_TEXT_COLOR);
        } else {
	        for (Rect r : facesArray)
	        {
	        	String tag = "";
	        	// For some reason the detector sometimes returns values outside
	        	// the video frame size range. Mat.submat does not like that at all.
            	if (r.tl().x < 0) break;
	        	if (r.tl().y < 0) break;
	        	if (r.br().x > mGray.width()-1) break;
	        	if (r.br().y > mGray.height()-1) break;
	        	
	        	Mat ROI = mGray.submat(r).clone();
            	int[] result = faceclass.identifyFace( ROI );
            	            	
	            if (faceclass.mIdentifierMethod != FaceDetectionAndProcessing.NO_IDENTIFICATION)
	            	if (result[0] > 0) tag = faceclass.identities.get(result[0]).getName();
	            	else tag = getResources().getString(R.string.unknown);
	            
	            tag += " is ";
	            
	            switch(result[1]) {
			    	case expressionclassifier.EMOTION_HAPPY:		tag += "happy";		break;
			    	case expressionclassifier.EMOTION_SAD:			tag += "sad";		break;
			    	//case expressionclassifier.EMOTION_SURPRISED:	tag += "surprised";	break;
			    	case expressionclassifier.EMOTION_ANGRY:		tag += "angry";		break;
			    	case expressionclassifier.EMOTION_DISGUSTED:	tag += "disgusted";	break;
			    	case expressionclassifier.EMOTION_AFRAID: 		tag += "afraid";	break;
			    	case expressionclassifier.EMOTION_NEUTRAL:		tag += "neutral";	break;
			    	default:tag += "badly expressed";
	            }
			
				switch(result[2]) {
					case expressionclassifier.GENDER_MALE:			tag += " male";		break;
					case expressionclassifier.GENDER_FEMALE:		tag += " female";	break;
					default:										tag += " android?";break;
				}
	         // Show the identity string under the rectangle
            	Size textsize = Core.getTextSize(tag, fontFace, fontScale, thickness, baseLine);
	        	
            	r.x += x_offset;
            	r.y += y_offset;
            	
            	if (result[2] == expressionclassifier.GENDER_MALE)
                	Core.rectangle(mRgba, r.tl(), r.br(), FACE_RECT_COLOR, 2);
            	else if (result[2] == expressionclassifier.GENDER_FEMALE) {
            		int size = (int)(r.br().x - r.tl().x)/2; // square area
            		Point cpoint = new Point( r.tl().x + size, r.tl().y + size );
            		
            		Core.circle(mRgba, cpoint , size, FACE_RECT_COLOR, 2);
            	} else {
            		Core.line(mRgba, r.tl(), r.br(), FACE_RECT_COLOR, 2);
            	}
            	
            	Core.rectangle(mRgba, new Point(r.tl().x, r.br().y+2),
            			new Point(r.tl().x+textsize.width, r.br().y+30), BLACK, Core.FILLED);
            			//new Point(facesArray[i].br().x, facesArray[i].br().y+30), BLACK, Core.FILLED);
            	Core.putText(mRgba, tag, new Point(r.tl().x, r.br().y+25), fontFace,
            			fontScale, NOTICE_TEXT_COLOR);
	        }
        }
        return mRgba;
    }

    @Override
    public boolean onCreateOptionsMenu(Menu menu) {
        Log.i(TAG, "called onCreateOptionsMenu");
        mItemSystem = menu.add("System");
        mItemTriggering = menu.add("Triggering");
        mItemRegocnition = menu.add("Recognition");
        mIdentification = menu.add("Identification");
        mItemPreprocessing = menu.add("Preprocessing");
        mItemTesting = menu.add("Testing");
        return true;
    }

    @Override
    public boolean onOptionsItemSelected(MenuItem item) {
    	AlertDialog.Builder builder = new AlertDialog.Builder(this);
    	CharSequence[] Sysmenu = {"Video resolution","Bounding box colour","Notice text color"};
    	CharSequence[] Triggermenu = {"Touch","Smile","Group smile","Spesific identity","New identity","Maximum rate"};
    	CharSequence[] Recognitionmenu = {"Viola-Jones","Local Binary Patterns"};
    	CharSequence[] Identificationmenu = {"Eigenfaces","Gabor Volume-LBP","Switch off identification"};
    	CharSequence[] Testingmenu = {"Photo gallery","Save Eigenface pictures","Calculate confusions","ID testing"};
    	CharSequence[] Preprocessingmenu = {"Turn off preprocessing","Histogram equalization", "Skin thresholding"};
        if (item == mItemSystem) {
    		builder.setTitle("System options");
    		builder.setItems(Sysmenu, new DialogInterface.OnClickListener() {
    			@Override public void onClick(DialogInterface dialog, int item) {
    		    	switch (item) {
    		    	case 0:
    		    		selectPreviewSize();
    		    		break;
    		    	case 1:
    		    		pickColour(0);
    		    		break;
    		    	case 2:
    		    		pickColour(1);
    		    		break;
    		    	default:break;
    		    	}}});
        } else if (item == mItemTriggering) {
        	builder.setTitle("Triggering options");
        	builder.setMultiChoiceItems(Triggermenu, selectedTriggerItems,
        			new DialogInterface.OnMultiChoiceClickListener(){
        		@Override public void onClick(DialogInterface dialog, int which, boolean isChecked) {
        			if (isChecked) {
        				selectedTriggerItems[which]=true;
        				switch(which) {
        				case SPESIFIC_TRIGGER:
        					break;

        				case NEWID_TRIGGER:
        					break;

        				case MAX_RATE_TRIGGER:
        					selectMaxTriggerRate();
        					break;
        				}
        			} else if (selectedTriggerItems[which]) {
        				selectedTriggerItems[which]=false;
        			}}});        	
        } else if (item == mItemRegocnition) {        	
    		builder.setTitle("Face recognition method");
    		builder.setItems(Recognitionmenu, new DialogInterface.OnClickListener() {
    		    @Override
    			public void onClick(DialogInterface dialog, int item) {
    		    	faceclass.selectFaceRecognicer( item );
    		    }});
    	} else if (item == mIdentification) {
    		builder.setTitle("Face identification method");
    		builder.setItems(Identificationmenu, new DialogInterface.OnClickListener() {
			@Override public void onClick(DialogInterface dialog, int item) {		    	
				switch (item) {		    	
		    	case 0:
		    		faceclass.mIdentifierMethod = FaceDetectionAndProcessing.EIGENFACES;
		    		retrainQuestion();
		    		break;
		    	case 1:
		    		faceclass.mIdentifierMethod = FaceDetectionAndProcessing.GVLBP;
		    		retrainQuestion();
		    		break;
		    	case 2:
		    		faceclass.mIdentifierMethod = FaceDetectionAndProcessing.NO_IDENTIFICATION;
		    		break;
		    	default:break;
		    	}
			}});
		} else if (item == mItemPreprocessing) {
        	builder.setTitle("Preprocessing");
        	builder.setMultiChoiceItems(Preprocessingmenu, selectedPreprocessor,
        			new DialogInterface.OnMultiChoiceClickListener(){
        		@Override public void onClick(DialogInterface dialog, int which, boolean isChecked) {
        			if (isChecked) {
        				selectedPreprocessor[which]=true;
        				switch(which) {
        				case NO_PREPROCESSING:
        					selectedPreprocessor[FULL_PIC_HISTOGRAMEQ]=false;
    						selectedPreprocessor[SKINCOLOUR_THRESHOLD]=false;
        					break;
        				case FULL_PIC_HISTOGRAMEQ:
        					selectedPreprocessor[NO_PREPROCESSING] = false;
        					break;

        				case SKINCOLOUR_THRESHOLD:
        					selectedPreprocessor[NO_PREPROCESSING] = false;
        					break;
        				}
        			} else if (selectedPreprocessor[which]) {
        				if (which != NO_PREPROCESSING) selectedPreprocessor[which]=false;
        			}
        			dialog.dismiss();
        		}});        	
        } else if (item == mItemTesting) {
    		builder.setTitle("Testing triggering");
    		builder.setItems(Testingmenu, new DialogInterface.OnClickListener() {
			@Override public void onClick(DialogInterface dialog, int item) {
				switch (item) {
				case 0:
		    		scanGallery();
		    		break;
		    	case 1:
		    		faceclass.saveEigenpictures();
		    		break;
		    	case 2:
		    		AlertDialog.Builder retrain = new AlertDialog.Builder(mainActivityContext);
		    		retrain.setTitle("Retrain attribute classifiers?");
		    		retrain.setPositiveButton("Yes", new DialogInterface.OnClickListener() {
		    			@Override
		    			public void onClick(DialogInterface dialog, int id) {
		    				faceclass.expclass.trainAttributes();
		    				faceclass.confusionMatrices();
		    			}
		    		});
		    		retrain.setNegativeButton("No", new DialogInterface.OnClickListener() {
		    			@Override
		    			public void onClick(DialogInterface dialog, int id) {
		    				faceclass.confusionMatrices();
		    			}		    			
		    		});
		    		
		    		AlertDialog retrainD = retrain.create();
		    		retrainD.show();
		    		
		    		break;
		    	case 3:
		    		faceclass.recognitionTest();
		    		break;
		    	default:break;
		    	}
				dialog.dismiss();
			}});
		}            
        
    		
		AlertDialog alert = builder.create();
		alert.show();
        
        return true;
    }
    
    private void retrainQuestion()
    {
    	AlertDialog.Builder builder = new AlertDialog.Builder(mainActivityContext);
    	String method;
    	switch (faceclass.mIdentifierMethod) {
    		case FaceDetectionAndProcessing.EIGENFACES: method = "Eigenfaces";	break;
    		case FaceDetectionAndProcessing.GVLBP: method = "Gabor Volumes LBP"; break;
    		default: method = "default switch - error"; break;    			
    	}
		builder.setTitle(method);
    	builder.setMessage("This will be a long operation. Do you want to retrain the classifier for this method?");
		builder.setNegativeButton("No", new DialogInterface.OnClickListener() { 
			@Override
			public void onClick(DialogInterface dialog, int id) {
				try {
					//loadClassifier();
				} catch (Exception e) {
					Toast.makeText(mainActivityContext, "Unable to open the classifier file(s)",
							Toast.LENGTH_LONG).show();
					return;
				}
				dialog.cancel();
			}
		});
		builder.setPositiveButton("Yes", new DialogInterface.OnClickListener() {
			@Override
			public void onClick(DialogInterface dialog, int id) {
				Toast.makeText(mainActivityContext, "Training.. This will take some time.",Toast.LENGTH_LONG).show();
				mOpenCvCameraView.disableView();
				
				switch (faceclass.mIdentifierMethod) {
				case FaceDetectionAndProcessing.EIGENFACES:
					faceclass.trainEigenfaceClassifier();
					break;
				case FaceDetectionAndProcessing.GVLBP:
					faceclass.trainGVLBPClassifier();
					break;
				default:break;
				}
				
				mOpenCvCameraView.enableView();
				dialog.cancel();
			}
		});
		AlertDialog retrainQuestionDialog = builder.create();
		retrainQuestionDialog.show();
    }
    
    
  protected void scanGallery() {
    	FragmentTransaction ft = getFragmentManager().beginTransaction();
        Fragment prev = getFragmentManager().findFragmentByTag("GalleryScanner");
        if (prev != null) ft.remove(prev);
        ft.addToBackStack(null);
        
        // Create and show the dialog.
        GalleryScanner newFragment = GalleryScanner.newInstance(faceclass, mOpenCvCameraView);
        newFragment.show(ft, "GalleryScanner");
	}

    private void selectPreviewSize() {
	    List<String> strlst = new ArrayList<String>();
		Iterator<Point> itr = desireable_previews.iterator();
		while(itr.hasNext()) {
			Point s = (Point)itr.next(); 
			String str = Integer.toString((int)s.x) + "x" + Integer.toString((int)s.y);
			strlst.add(str);
		}
	    final CharSequence[] items = strlst.toArray(new String[strlst.size()]);
	    
		AlertDialog.Builder builder = new AlertDialog.Builder(this);
		builder.setTitle("Select video resolution");
		builder.setItems(items, new DialogInterface.OnClickListener() {
		    public void onClick(DialogInterface dialog, int item) {
		    	Point des = desireable_previews.get(item);
		    	mOpenCvCameraView.disableView();
		    	mOpenCvCameraView.setMaxFrameSize((int)des.x+2, (int)des.y+2);
		    	photoSize = des;
		    	
		    	if (faceclass.mAbsoluteFaceSize == 0) {		            
		            if (Math.round(des.y * faceclass.mRelativeFaceSize) > 0) {
		                faceclass.mAbsoluteFaceSize = (int)Math.round(des.y * faceclass.mRelativeFaceSize);
		            }
		            faceclass.mNativeDetector.setMinFaceSize(faceclass.mAbsoluteFaceSize);
		        }
		    	// Magic numbers.. The system will pick the resolution as high
		    	// as possible so those 8's are there to ensure that the picked
		    	// resolution is chosen. Without them there was some odd results.
		    	mOpenCvCameraView.enableView();
		    }
		});
		AlertDialog alert = builder.create();
		alert.show();
    }
    
    private void selectMaxTriggerRate() {
	    final Dialog dialog1 = new Dialog(mainActivityContext);
		dialog1.setContentView(R.layout.maxrate_spinner);
		dialog1.setTitle("Maximum picture rate");
	
		// set the custom dialog components - text, image and button
		final TextView text = (TextView)dialog1.findViewById(R.id.textView_distance);
		text.setText(R.string.maxrate_prompt);
		final TextView text2 = (TextView)dialog1.findViewById(R.id.textView_expression);
		text2.setText( maxPhotoRate+"s");
		final Button but = (Button)dialog1.findViewById(R.id.button1);
		but.setText("Accept");
		
		SeekBar seekBar = (SeekBar)dialog1.findViewById(R.id.seekBar1);
	    seekBar.setMax(120);
	    seekBar.setProgress(maxPhotoRate);
	    seekBar.setOnSeekBarChangeListener(new OnSeekBarChangeListener() {
	    	@Override
		    public void onStopTrackingTouch(SeekBar seekBar) {
		    }            	    	   
		    @Override
		    public void onStartTrackingTouch(SeekBar seekBar) {
		    }            	    	   
		    @Override
		    public void onProgressChanged(SeekBar seekBar, int progress,
		     boolean fromUser) {
		    	
			    text2.setText(progress+" s");
			    maxPhotoRate = progress;
		   }
	    });
		but.setOnClickListener(new OnClickListener() {
			@Override
			public void onClick(View v) {
				dialog1.dismiss();
			}
		});
		dialog1.show();
    }
    
    private void pickColour(int which)
    {
    	final Dialog colordialog = new Dialog(mainActivityContext);
    	colordialog.setContentView(R.layout.colorpicker);
		
		final Scalar color;
		if (which==0) {
			colordialog.setTitle("Bounding box color");
			color = FACE_RECT_COLOR;
		} else {
			colordialog.setTitle("On screen text color");
			color = NOTICE_TEXT_COLOR;
		}
		
		final Button OKbut = (Button)colordialog.findViewById(R.id.button1);		
		final TextView colText = (TextView)colordialog.findViewById(R.id.textView_age);
		
		colText.setTextColor(Color.rgb((int)color.val[0], (int)color.val[1], (int)color.val[2]));
		
		SeekBar redSeekBar = (SeekBar)colordialog.findViewById(R.id.redSeekBar);
	    SeekBar greenSeekBar = (SeekBar)colordialog.findViewById(R.id.greenSeekBar);
	    SeekBar blueSeekBar = (SeekBar)colordialog.findViewById(R.id.blueSeekBar);
	    
	    //Scalars internally doubles.. what a brilliant type..
		redSeekBar.setMax(255); redSeekBar.setProgress((int)color.val[0]);
		greenSeekBar.setMax(255); greenSeekBar.setProgress((int)color.val[1]);
		blueSeekBar.setMax(255); blueSeekBar.setProgress((int)color.val[2]);
		
	    redSeekBar.setOnSeekBarChangeListener(new OnSeekBarChangeListener() {
	    	@Override public void onStopTrackingTouch(SeekBar seekBar) { }            	    	   
		    @Override public void onStartTrackingTouch(SeekBar seekBar) { }            	    	   
		    @Override public void onProgressChanged(SeekBar seekBar, int progress,
		     boolean fromUser) {
		    	 color.val[0] = progress;
		    	 colText.setTextColor(Color.rgb((int)color.val[0], (int)color.val[1], (int)color.val[2]));
			 }
	    });
	    greenSeekBar.setOnSeekBarChangeListener(new OnSeekBarChangeListener() {
	    	@Override public void onStopTrackingTouch(SeekBar seekBar) { }            	    	   
		    @Override public void onStartTrackingTouch(SeekBar seekBar) { }            	    	   
		    @Override public void onProgressChanged(SeekBar seekBar, int progress,
		     boolean fromUser) {
		    	 color.val[1] = progress;
		    	 colText.setTextColor(Color.rgb((int)color.val[0], (int)color.val[1], (int)color.val[2]));
			 }
	    });
	    blueSeekBar.setOnSeekBarChangeListener(new OnSeekBarChangeListener() {
	    	@Override public void onStopTrackingTouch(SeekBar seekBar) { }            	    	   
		    @Override public void onStartTrackingTouch(SeekBar seekBar) { }            	    	   
		    @Override public void onProgressChanged(SeekBar seekBar, int progress,
		     boolean fromUser) {
		    	 color.val[2] = progress;
		    	 colText.setTextColor(Color.rgb((int)color.val[0], (int)color.val[1], (int)color.val[2]));
			 }
	    });
	    
		OKbut.setOnClickListener(new OnClickListener() {
			@Override
			public void onClick(View v) {
				colordialog.dismiss();
			}
		});
		colordialog.show();
    }
}


    