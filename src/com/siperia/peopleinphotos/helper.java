package com.siperia.peopleinphotos;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileOutputStream;
import java.io.FileWriter;
import java.text.SimpleDateFormat;
import java.util.Calendar;
import java.util.List;
import java.util.Random;
import java.util.Map.Entry;

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
    
	static void savePicture(Mat mat, Bitmap bmap, boolean toast, String prefix) {
    	Random generator2 = new Random( System.currentTimeMillis() );
    	
    	if (mat != null && bmap == null) {
	    	if (mat.type() != CvType.CV_8U) {
	    		mat = mat.clone();
	    		Core.normalize(mat, mat, 0, 255, Core.NORM_MINMAX);
	    		mat.convertTo(mat, CvType.CV_8U);
	    	}
	    	
	    	bmap = Bitmap.createBitmap(mat.width(), mat.height(), Bitmap.Config.ARGB_8888);
	    	Utils.matToBitmap(mat, bmap);
    	}
    	
    	if (bmap != null)
    	{
    		if (!pictureFileDir.exists() && !pictureFileDir.mkdirs()) {
    			return;
    		}
    		
    		Calendar c = Calendar.getInstance();
    		
    		String photoFile = "Pic_" + prefix + generator2.nextInt() +".png";
    		Log.d(TAG, "writing:"+photoFile);
    		String filename = pictureFileDir.getPath() + File.separator + photoFile;

    		File pictureFile = new File(filename);
    		try {
    			FileOutputStream fos = new FileOutputStream(pictureFile);
    			bmap.compress(Bitmap.CompressFormat.valueOf("PNG") , 100, fos);
    			fos.close();
    		} catch (Exception error) {
    			Log.e(TAG,"bad picture save");
    		}
    	}
    }
	
	@SuppressWarnings("unused")
	static String FtoString(float[] floats) {
		String s = "";
		for (int g=0;g<floats.length;g++) {
			s += Float.toString(floats[g]) + " ";
		}
		return s;
	}
    
	@SuppressWarnings("unused")
	static String DtoString(double[] tn) {
		String s = "";
		for (int g=0;g<tn.length;g++) {
			s += Double.toString(tn[g]) + " ";
		}
		return s;
	}

	@SuppressWarnings("unused")
	static String ItoString(int[] ints) {
		String s = "";
		for (int g=0;g<ints.length;g++) {
			s += Integer.toString(ints[g]) + " ";
		}
		return s;
	}
	
	static void shuffleFileArray(File[] ar) {
	    Random rnd = new Random();
	    for (int i = ar.length - 1; i > 0; i--)
	    {
	      int index = rnd.nextInt(i + 1);
	      File a = ar[index];
	      ar[index] = ar[i];
	      ar[i] = a;
	    }
	}
	
	static void dumpLine( String filename, String line ) {
		try {
			File file = new File(filename);
			if (!file.exists()) file.createNewFile();
			
			FileWriter fw = new FileWriter(file, true);
			BufferedWriter bw = new BufferedWriter(fw);
			
			bw.write( line );
			bw.newLine();

			bw.close();
			fw.close();
		} catch (Exception e) {
			e.printStackTrace();
		}
	}
	
	static void savePair( Mat p1, Mat p2, String prefix ) {
		int height = Math.max((int)p1.size().height, (int)p2.size().height);
		Mat result = new Mat( height, (int)(p1.size().width+p2.size().width), CvType.CV_8U );
	
		p1.copyTo( result.colRange(0, (int)p1.size().width) );
		p2.copyTo( result.colRange( (int)p1.size().width, result.width() ) );
	
		savePicture( result, null, false, prefix );
	}
	
    static void crash() { Core.divide(new Mat(), new Mat(), null); }
}

