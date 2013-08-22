package com.siperia.peopleinphotos;

import java.util.List;

import android.app.DialogFragment;
import android.os.Bundle;
import android.support.v4.view.PagerAdapter;
import android.support.v4.view.ViewPager;
import android.view.LayoutInflater;
import android.view.Menu;
import android.view.View;
import android.view.ViewGroup;

public class showMatch_fragment extends DialogFragment {
	
	private View		view;
	
	public showMatch_fragment() {super();}
	
	ViewPager			viewPager;
    PagerAdapter		adapter;
    List<Similarity>	sims;
    int					reqW,reqH;
 
    @Override
    public void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
    }
    
    @Override
    public View onCreateView(LayoutInflater inflater, ViewGroup container,
            Bundle savedInstanceState) {
    	
    	view = inflater.inflate(R.layout.viewpager_fragment, container);
    	
    	// Get the view from viewpager_main.xml
        //setContentView(R.layout.viewpager_fragment);
    	getDialog().setTitle("Closest matches");
 
        // Locate the ViewPager in viewpager_main.xml
        viewPager = (ViewPager)view.findViewById(R.id.pager);
        // Pass results to ViewPagerAdapter Class
        adapter = new ViewPagerAdapter(getActivity(), sims, reqW, reqH);
        // Binds the Adapter to the ViewPager
        viewPager.setAdapter(adapter);
 
        return view;
    }
    
    public void setArgs( List<Similarity> sims, int reqW, int reqH ) {
    	this.sims = sims;
    	this.reqW = reqW;
    	this.reqH = reqH;
    }
 
    // Not using options menu in this tutorial
    public boolean onCreateOptionsMenu(Menu menu) {
        //getActivity().getMenuInflater().inflate(R.menu.activity_main, menu);
        return true;
    }

}
