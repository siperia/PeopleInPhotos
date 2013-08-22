package com.siperia.peopleinphotos;

import java.io.File;
import java.util.List;

import android.content.Context;
import android.graphics.Bitmap;
import android.support.v4.view.PagerAdapter;
import android.support.v4.view.ViewPager;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.ImageView;
import android.widget.RelativeLayout;
import android.widget.TextView;

public class ViewPagerAdapter extends PagerAdapter {
    // Declare Variables
    Context context;
    List<Similarity> sims;
    int reqWidth, reqHeight;
    
    LayoutInflater inflater;
 
    public ViewPagerAdapter(Context context, List<Similarity> sims_in, int reqW, int reqH ) {
        this.context = context;
    	this.sims = sims_in;
        this.reqWidth = reqW;
        this.reqHeight = reqH;
    }
 
    @Override
    public int getCount() {
        return sims.size();
    }
 
    @Override
    public boolean isViewFromObject(View view, Object object) {
        return view == ((View)object); // == ((RelativeLayout) object);
    }
 
    @Override
    public Object instantiateItem(ViewGroup container, int position) {
 
        // Declare Variables
        ImageView imgview;
        TextView txtscore;
        
        inflater = (LayoutInflater) context.getSystemService(Context.LAYOUT_INFLATER_SERVICE);
        View itemView = inflater.inflate(R.layout.viewpager_item, container,false);
         
        // Locate the TextViews in viewpager_item.xml
        txtscore = (TextView) itemView.findViewById(R.id.matchscore);
        // Capture position and set to the TextViews
        txtscore.setText(Integer.toString((int)sims.get(position).score));
         
        // Locate the ImageView in viewpager_item.xml
        imgview = (ImageView)itemView.findViewById(R.id.imageview);
        // Capture position and set to the ImageView
        
        // Load photo and display it in the imageview
        Bitmap bmap = GalleryScanner.decodeSampledBitmapFromFile(new File(sims.get(position).string), reqWidth, reqHeight);
                
        imgview.setImageBitmap(bmap);
 
        // Add viewpager_item.xml to ViewPager
        ((ViewPager) container).addView(itemView);
        return itemView;
    }
 
    @Override
    public void destroyItem(ViewGroup container, int position, Object object) {
        // Remove viewpager_item.xml from ViewPager
        ((ViewPager) container).removeView((View) object);
    }
}
