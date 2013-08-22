package com.siperia.peopleinphotos;

import java.io.File;
import java.io.FileOutputStream;
import java.text.SimpleDateFormat;
import java.util.Calendar;
import java.util.Random;

import org.opencv.android.Utils;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfDouble;

import android.graphics.Bitmap;
import android.os.Environment;
import android.util.Log;
import android.widget.Toast;

public class helper {
	private static final String		TAG="Helper";
	private static final File 	   sdDir = Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_PICTURES);
    private static final File	   identRootDir			= new File(sdDir, "PiP_idents");
    private static final File	   pictureFileDir		= new File(sdDir, "PiP");
    
	static void savePicture(Mat mat, boolean toast, String prefix) {
    	Bitmap.Config conf = Bitmap.Config.ARGB_8888;
    	Random generator2 = new Random( System.currentTimeMillis() );
    	
    	Bitmap bmp = Bitmap.createBitmap(mat.width(), mat.height(), conf);
    	Utils.matToBitmap(mat, bmp);
    	
    	if (bmp != null)
    	{
    		if (!pictureFileDir.exists() && !pictureFileDir.mkdirs()) {
    			return;
    		}
    		
    		Calendar c = Calendar.getInstance();
    		
    		SimpleDateFormat df3 = new SimpleDateFormat("dd-MM-yyyy_HH-mm-ss");
    		String formattedDate = df3.format(c.getTime());
    		
    		String photoFile = "Pic_" + prefix + formattedDate + generator2.nextInt() +".jpg";
    		Log.d(TAG, "writing:"+photoFile);
    		String filename = pictureFileDir.getPath() + File.separator + photoFile;

    		File pictureFile = new File(filename);
    		try {
    			FileOutputStream fos = new FileOutputStream(pictureFile);
    			bmp.compress(Bitmap.CompressFormat.valueOf("JPEG") , 100, fos);
    			fos.close();
    		} catch (Exception error) {
    			Log.e(TAG,"bad picture save");
    		}
    	}
    }
	
    static void crash() { Core.divide(new Mat(), new Mat(), null); }
}

