package com.siperia.peopleinphotos;

import java.util.ArrayList;
import java.util.List;

import org.opencv.core.Mat;
import org.opencv.core.Point;
import org.opencv.core.Rect;

import android.util.Pair;
import delaunay_triangulation.Point_dt;

// describes a face in a photo
public class face {
	
	public face (int[] a, Rect locationInPhoto, float[] gaborVect) {
		this.attributes = a.clone();
		this.location = locationInPhoto.clone();
		this.gv = gaborVect.clone();
	}
	
	public int[]		attributes;
	public float[]		gv;
	Rect 				location;
	Mat					fullPic;
	
	public Pair<Point_dt,Point_dt> getDTrect () {
		return new Pair<Point_dt,Point_dt>(new Point_dt(location.x,location.y), new Point_dt(location.width,location.height));
	}
	
	public void setFullPic( Mat pic ) { fullPic = pic.clone(); }
	public Mat getFullPic() { return fullPic; };

	public int[] getAttributes() {
		return attributes;
	}
		
	public Point_dt rectCenter_dt() {
		return new Point_dt(location.x + (location.width / 2), location.y + (location.height / 2));
	}
	
	public Point rectCenter() {
		return new Point(location.x + (location.width / 2), location.y + (location.height / 2));
	}
}
