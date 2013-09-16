package com.siperia.peopleinphotos;

import java.io.File;
import java.io.FileOutputStream;
import java.util.Random;

import org.opencv.android.Utils;
import org.opencv.core.Mat;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;

import android.app.Dialog;
import android.app.DialogFragment;
import android.content.DialogInterface;
import android.graphics.Bitmap;
import android.os.Bundle;
import android.os.Environment;
import android.text.Editable;
import android.util.Log;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.view.View.OnClickListener;
import android.widget.ArrayAdapter;
import android.widget.Button;
import android.widget.EditText;
import android.widget.ImageView;
import android.widget.Spinner;
import android.widget.TextView;
import android.widget.Toast;

public class addUser_fragment extends DialogFragment {
	
	private static final String    TAG					= "PeopleInPhotos::MainActivity";
    private static final File 	   sdDir = Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_PICTURES);
    private static final File	   identRootDir			= new File(sdDir, "PiP_idents");
    private static final File	   pictureFileDir		= new File(sdDir, "PiP");
    
	private View view;
	private ImageView image;
	
	private Mat ROI = null;
	private Bitmap facepic;
	private Bitmap thumbnail;
	private int i;
	private int t;
	private FaceDetectionAndProcessing faceclass = null;

	public addUser_fragment() {super();}
	
	public void setArgs(Mat ROI_, int i_, int t_, FaceDetectionAndProcessing face_) {
		ROI = ROI_;
		
		facepic = Bitmap.createBitmap(ROI.width(), ROI.height(), Bitmap.Config.ARGB_8888);
		Utils.matToBitmap(ROI, facepic);
		
		Mat tmpmat = new Mat();
		Imgproc.resize(ROI, tmpmat, new Size(150,150));
		thumbnail = Bitmap.createBitmap(150,150, Bitmap.Config.ARGB_8888);
		Utils.matToBitmap(tmpmat, thumbnail);
		
		i = i_;
		t = t_;
		faceclass = face_;
	}
	
	@Override
	public void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
	}
	
	@Override
    public View onCreateView(LayoutInflater inflater, ViewGroup container,
            Bundle savedInstanceState) {
		
		view = inflater.inflate(R.layout.teacherdialog, container);
		getDialog().setTitle("Add new teaching sample ("+(t-i)+"/"+t+")");
		
		image = (ImageView)view.findViewById(R.id.imageView1);
				
		image.setImageBitmap(thumbnail);
		
		final TextView text = (TextView)view.findViewById(R.id.textView_distance);
		text.setText(R.string.whoisthis);
		final TextView text2 = (TextView)view.findViewById(R.id.textView_expression);
		text2.setText(R.string.ithink);
		
		final Button but = (Button)view.findViewById(R.id.button1);
		but.setText(R.string.noone);
		but.setOnClickListener(new OnClickListener() {
			@Override
			public void onClick(View v) {
				dismiss();
			}
		});
		
		final Button but2 = (Button)view.findViewById(R.id.button2);
		but2.setText(R.string.someonenew);
		but2.setOnClickListener(new OnClickListener() {
			@Override
			public void onClick(View v) {
				final Dialog identdialog = new Dialog(getActivity());
				identdialog.setContentView(R.layout.addidentity);
				identdialog.setTitle("New name for an identity");
				
				final Button OKbut = (Button)identdialog.findViewById(R.id.OK);
				final EditText edi = (EditText)identdialog.findViewById(R.id.editText1);
				OKbut.setOnClickListener(new OnClickListener() {
					@Override
					public void onClick(View v) {
						Editable name = edi.getText();
						if (name.length() > 0)
						{
							boolean existingID = false;
							for (Identity ID: faceclass.identities) {
								if (ID.getName().equals( name.toString() )) existingID = true;
							}
							if (!existingID) {
								int max = -1; // first = 0
								for (Identity ID: faceclass.identities)
									if (ID.getID() > max) max = ID.getID();
								
								Identity newIdent = new Identity(name.toString(), max+1);
								faceclass.identities.add( newIdent );
								storeFaceSample( name.toString(), facepic );
								Toast.makeText(getActivity(), "Identity "+name.toString() + " added.",
				    					Toast.LENGTH_LONG).show();
								faceclass.updateSampleFiles();
								identdialog.dismiss();
								dismiss();
							} else {
								Toast.makeText(getActivity(), "Identity "+name.toString() + " allready exists.",
				    					Toast.LENGTH_LONG).show();
							}
						}
					}
				});
				
				identdialog.show();
			}
		});
		
		final Spinner spin = (Spinner)view.findViewById(R.id.spinner1);
		// Create an ArrayAdapter using the string array and a default spinner layout
		ArrayAdapter<String> adapter = new ArrayAdapter<String>(getActivity(),
				android.R.layout.simple_spinner_item , faceclass.getIdents());
		adapter.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item);
		// Apply the adapter to the spinner
		spin.setAdapter(adapter);
		
		final Button okButton = (Button)view.findViewById(R.id.OK);
		okButton.setText(R.string.OK);
		okButton.setOnClickListener(new OnClickListener() {
			@Override
			public void onClick(View v) {
				if (spin.getCount() > 0) {
					String ident = (String)spin.getSelectedItem();
					storeFaceSample( ident, facepic );
					dismiss();
				} else {
					Toast.makeText(getActivity(), "There are no known identities yet.",
	    					Toast.LENGTH_LONG).show();						
				}
			}
		});
		
		return view;
	}
	
	// Saves a sample image taken from frame grab. (or anywhere else)
	public void storeFaceSample( String ident, Bitmap pic) {
		
		Random generator = new Random( System.currentTimeMillis() );
		try {
    		if (!identRootDir.exists() && !identRootDir.mkdirs()) {
    			throw new Exception();
    		}
		} catch ( Exception  e ) {				    			
			Toast.makeText(getActivity(), "Can't open directory to store identity photos.",
					Toast.LENGTH_LONG).show();
			return;
		}
		
		String photoFileName;
		String absoluteFileName = null;
		
		int classNo=99999;
		for (Identity ID: faceclass.identities) if (ID.getName().equals(ident)) classNo = ID.getID();
							
		photoFileName = ident + "_" + classNo + "_" + generator.nextInt() + ".JPG";
		absoluteFileName = identRootDir.getPath() + File.separator + photoFileName;
		//Log.d(TAG, "Probing existance of file "+absoluteFileName);
				
		File photoFile = new File(absoluteFileName);
		try {
			FileOutputStream fos = new FileOutputStream(photoFile);
			pic.compress(Bitmap.CompressFormat.JPEG, 100, fos);
			fos.close();
		} catch (Exception error) {
			Toast.makeText(getActivity(), "Couldn't save the ident sample.", Toast.LENGTH_LONG).show();
		}
	}
}
