package com.siperia.peopleinphotos;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileReader;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;
import java.util.Scanner;

import org.opencv.android.Utils;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfFloat;
import org.opencv.core.MatOfInt;
import org.opencv.core.MatOfRect;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;

import com.siperia.peopleinphotos.Identity.sample;

import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.BitmapRegionDecoder;
import android.os.Environment;
import android.util.Log;
import android.util.Pair;
import delaunay_triangulation.BoundingBox;
import delaunay_triangulation.Delaunay_Triangulation;
import delaunay_triangulation.Point_dt;
import delaunay_triangulation.Triangle_dt;

public class photo {
	private static final String		TAG = "PeopleInPhotos::photo";
	private static final File		sdDir = Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_PICTURES);
	private static final File		identRootDir = new File(sdDir, "PiP_idents");
	private static final String		photoInfoFile = identRootDir.getAbsolutePath() + "/DCIMdata.txt";
	public Size 					maxFacesSize = new Size(20,20);
	
	String 							path;
	Bitmap							bitmap = null;
	Mat								matpic = null;
	Mat								gray = null;
	
	int 							picWidth,picHeight;
	
	MatOfRect						faces_mor = new MatOfRect();
	public List<face> 				faces = new ArrayList<face>();
		
	boolean 						found = false;
	
	List<Pair<Point_dt, Point_dt>> 	compRNG = new ArrayList<Pair<Point_dt, Point_dt>>();
	List<Pair<Point_dt, Point_dt>> 	RNG = new ArrayList<Pair<Point_dt, Point_dt>>();
	List<Pair<Point_dt, Point_dt>> 	notRNG = new ArrayList<Pair<Point_dt, Point_dt>>();
	
	Mat 							adj = null;
		
	FaceDetectionAndProcessing		fc = null;
	
	public photo(String this_file, FaceDetectionAndProcessing faceclass, boolean find) {
		
		// Reads previously classified image data from photoInfoFile-file.
		this.path = this_file;
		fc = faceclass;
		if (find) {
			try		
	        {
	            FileReader fileReader = new FileReader(new File(photoInfoFile));
	            Scanner scan = new Scanner( fileReader );
	            
	            while (!found && scan.hasNext()) {
	            	String fname = scan.next();
	            	if (fname.equals(this_file)) found = true; else scan.nextLine();
	            }
	            if (found) {
		            int facesToCompare = scan.nextInt();
		            float[] gaborVect = null;
		            	            
		            int[] toadd = new int[4];
		        	for (int face=0;face < facesToCompare;face++)
		        	{
		        		int x=scan.nextInt();
		        		int y=scan.nextInt();
		        		int w=scan.nextInt();
		        		int h=scan.nextInt();
		        		
		        		toadd[expressionclassifier.INDEX_IDENTITY] = 0; //placeholder
		        		toadd[expressionclassifier.INDEX_EXPRESSION] = scan.nextInt();
		        		toadd[expressionclassifier.INDEX_GENDER] = scan.nextInt();
		        		toadd[expressionclassifier.INDEX_AGE] = scan.nextInt();
		        		
		        		int gabors = scan.nextInt();
		        		if (gaborVect == null) gaborVect = new float[gabors];
		        		
		        		for (int j=0;j<gabors;j++) gaborVect[j] = scan.nextFloat();
		        		
		        		face newFace = new face(toadd, new Rect(x,y,w,h), gaborVect );
		        		faces.add(newFace);
		        	}
		        	
		        	int graphEdges = scan.nextInt();
		        	RNG.clear();
		        	for (int edge=0;edge<graphEdges;edge++) {
		        		RNG.add(new Pair<Point_dt, Point_dt>(new Point_dt(scan.nextDouble(),scan.nextDouble()),new Point_dt(scan.nextDouble(),scan.nextDouble())));
		        	}
		        	
		        	adj = Mat.zeros(maxFacesSize, CvType.CV_32FC1);
		        	for (int i=0;i<graphEdges;i++) {
		        		for (int j=0;j<graphEdges;j++) {
		        			adj.put(i, j, scan.nextInt());
		        		}
		        		Log.d(TAG, "debug: adj: "+adj.row(i).dump());
		        	}
		    		//Log.d(TAG, compEdgeMatrix.dump());
	            } 
	            scan.close();
	        }
	        catch ( Exception e )
	        {
	            e.printStackTrace();
	        }
		} else {
        	load();
        	detectFaces();
        	createRNG();
        	calcAdjacency();
        }
		
	}
	
	public Bitmap load() {
		// First decode with inJustDecodeBounds=true to check dimensions
        final BitmapFactory.Options ops = new BitmapFactory.Options();
        ops.inJustDecodeBounds = true;
        BitmapFactory.decodeFile(path,ops);

        // Calculate inSampleSize
        ops.inSampleSize = calculateInSampleSize(ops, 2000, 1000);
        // Decode bitmap with inSampleSize set
        ops.inJustDecodeBounds = false;
        // Needed for canvas editing
        ops.inMutable = true;        
        
		bitmap = BitmapFactory.decodeFile(path,ops);
		Log.d(TAG, "Loaded sized:"+bitmap.getHeight()+"x"+bitmap.getWidth());
		return bitmap;
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
	
	public void detectFaces( ) {
		matpic = new Mat();
		gray = new Mat();
		MatOfInt hist = new MatOfInt();
		
		faces.clear();
		
		Utils.bitmapToMat(bitmap, matpic);
		Imgproc.cvtColor(matpic, gray, Imgproc.COLOR_RGB2GRAY);
		Imgproc.equalizeHist(gray, gray);
		
		picWidth = gray.width();
		picHeight = gray.height();
		
		fc.mCascadeClassifier.detectMultiScale(gray, faces_mor, 1.1, 3, 0, new Size(64,64), gray.size());
						
		List<Rect> facelist = faces_mor.toList();
		for (Rect l: facelist) {
			MatOfRect result = new MatOfRect();
			
			Mat facepic = matpic.submat(l).clone();
			fc.expclass.nativelib.skinThreshold(facepic, result, false); // just the pixel count
							
			double skin = (double)result.toList().get(1).x / l.area();
			if ( skin > 0.4 ) { // enough "skin" to be a face
				Mat grayface = gray.submat(l).clone();
				
				int[] props = fc.identifyFace(grayface);
				fc.expclass.nativelib.GaborLBP_Histograms(grayface, hist, fc.LUT, 8, -1, -1);
				
	        	float[] gaborVals = new MatOfFloat( hist ).toArray();
				
				faces.add( new face(props, l, gaborVals)  );
			}
		}
	}
	
	public void getFullSizedSamples() {
		try {
			FileInputStream is = new FileInputStream(path);
						
			BitmapRegionDecoder brd = BitmapRegionDecoder.newInstance(is, true);
			BitmapFactory.Options opts = new BitmapFactory.Options();
			opts.outHeight = 200;
			opts.outWidth = 200;
			
			for (face f: faces) {
				Mat thumbpic = new Mat();
				android.graphics.Rect androidRect = new android.graphics.Rect(
						(int)f.location.tl().x, (int)f.location.tl().y,
						(int)f.location.br().x, (int)f.location.br().y );
								
				Bitmap bit_in = brd.decodeRegion(androidRect, opts);
				Utils.bitmapToMat(bit_in, thumbpic);
				
				//helper.savePicture(matpic, false, "full");
				f.setFullPic( thumbpic );
			}
			is.close();
		} catch ( Exception e ) {
			e.printStackTrace();
		}		
	}
	
	// scale the locations in this photo to match the given scale
    protected void scaleToMean(double scaleTo) {
    	
    	if (faces.size() == 0) return;
    	
    	double comp_mean_face_size = 1;
    	if (scaleTo != 1) {
	    	for (face f: faces) {
	    		comp_mean_face_size += f.location.width;
	    	}
	    	comp_mean_face_size /= faces.size();
    	}
    	
    	double scaling_factor = scaleTo / comp_mean_face_size;
    	//Log.d(TAG, "scaling_factor = "+scaling_factor);
    	
    	double meanX=0,meanY=0;
    	for (int i=0;i<faces.size();i++) {
    		meanX += faces.get(i).location.x;
    		meanY += faces.get(i).location.y;
    	}
    	meanX /= faces.size();
    	meanY /= faces.size();
    	
    	// rounding to integers won't matter that much
    	for (face f: faces) {
    		// remove mean from locations
    		f.location.y = (int)(f.location.y - meanY);
    		f.location.x = (int)(f.location.x - meanX);
    		    		
    		// scale the face sizes with scaling factor
    		f.location.width = (int)( f.location.width * scaling_factor );
    		f.location.height = (int)( f.location.height * scaling_factor );    		
    	}
    }
	
	public void createRNG() {
		Delaunay_Triangulation DTri = new Delaunay_Triangulation();
		RNG.clear();
		notRNG.clear();
		
		if (faces.size() > 2) {
			for (int i=0; i < faces.size(); i++) {
				Point face = faces.get(i).rectCenter();
				Point_dt dt = new Point_dt(face.x, face.y);
				
				DTri.insertPoint(dt);
			}
			
			Iterator<Triangle_dt> T_iter = DTri.trianglesIterator();
			
			// dirty full triangulation extractor for screenshot example, lot overlapping edges
			/*while (T_iter.hasNext()) {
				Triangle_dt tri = T_iter.next();
				RNG.add(new Pair<Point_dt,Point_dt>(tri.p1(),tri.p2()));
				RNG.add(new Pair<Point_dt,Point_dt>(tri.p2(),tri.p3()));
				RNG.add(new Pair<Point_dt,Point_dt>(tri.p3(),tri.p1()));
			}*/
			
			while (T_iter.hasNext()) {
				Triangle_dt triangle = T_iter.next();
				
				// reject triangles with vertices outside the frame as these
				// are the "support" triangles forming the convex outer rim.
				if (triangle.p1() != null && triangle.p2() != null && triangle.p3() != null) {
					BoundingBox bb = triangle.getBoundingBox();
					if (bb.minX() >= 0 && bb.minY() >= 0 && bb.maxX() < bitmap.getWidth() && bb.maxY() < bitmap.getHeight()) {
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
		} else if (faces.size() == 2) {
			Point face1 = faces.get(0).rectCenter(); Point_dt f1 = new Point_dt(face1.x,face1.y);
			Point face2 = faces.get(1).rectCenter(); Point_dt f2 = new Point_dt(face2.x,face2.y);
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
		
		Log.d(TAG, "size of triangle:"+DTri.trianglesSize()+", RNG:"+RNG.size()+", size of bad:"+notRNG.size());
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
			if (bb.minX() >= 0 && bb.minY() >= 0 && bb.maxX() < bitmap.getWidth() && bb.maxY() < bitmap.getHeight()) {
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
	
	public void calcAdjacency( ) {		
		adj = Mat.zeros(maxFacesSize, CvType.CV_32FC1);
		double match_distance = 2; // *shrug* small enough for possible roundings
		
		// catches everything twice but it doesn't matter as the links are symmetrical
		for (int edge = 0; edge<RNG.size(); edge++) {
			int face1_ind=Integer.MAX_VALUE, face2_ind=Integer.MAX_VALUE;
			
			for (int f1=0;f1<faces.size();f1++) {
				if ( faces.get(f1).rectCenter_dt().distance( RNG.get(edge).first ) < match_distance ) {
					face1_ind = f1;
				} else if ( faces.get(f1).rectCenter_dt().distance( RNG.get(edge).second ) < match_distance ) {
					face2_ind = f1;
				}
			}
			
			if ((face1_ind != Integer.MAX_VALUE) && (face2_ind != Integer.MAX_VALUE)) {				
				adj.put(face1_ind, face2_ind, 1); // these 2 edges are adjacent
				adj.put(face2_ind, face1_ind, 1); //and symmetry
			}
        }
	}
	
	// This is used to somehow decide best similarity between 1 face from this photo and all others
	// from "compare"
	public Mat matchFaces( photo compare ) {
		int matsize = ( this.faces.size() > compare.faces.size() ) ? this.faces.size() : compare.faces.size();
		Mat ret = Mat.zeros( new Size(matsize, matsize), CvType.CV_32FC1 );
		
		double[][] matches = new double[this.faces.size()][compare.faces.size()];
		
		Mat hist = new Mat( 256, 1, CvType.CV_32F );
		Mat comp_hist = new Mat( 256, 1, CvType.CV_32F );
		String s = "";
		
		for (int i = 0; i<this.faces.size();i++) {
			fc.expclass.nativelib.GaborLBP_Histograms(this.faces.get(i).getFullPic().clone(), hist, fc.LUT, 8, -1, -1);
			
			for (int j = 0; j<compare.faces.size();j++) {
				fc.expclass.nativelib.GaborLBP_Histograms(compare.faces.get(i).getFullPic().clone(), comp_hist, fc.LUT, 8, -1, -1);
				
				matches[i][j] = Imgproc.compareHist(hist, comp_hist, Imgproc.CV_COMP_CORREL);
				s += matches[i][j] + " ";
			}
			s += "\n";
		}
		
		Log.d(TAG, "debug: gabor correlations: "+ s);
		
		Hungarian hungarian = new Hungarian( matches );
		int[] selected = hungarian.execute();
		
		// set the selected indices to 1 in the permutation matrix to indicate a match
		for (int i = 0; i<selected.length; i++) {
			ret.put(i, selected[i], 1);
		}
		
		Log.d(TAG, "debug: permutation = "+ret.dump());
		
		return ret;
	}
	

}
