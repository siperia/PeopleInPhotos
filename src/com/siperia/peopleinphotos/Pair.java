package com.siperia.peopleinphotos;

public class Pair<F, S> {
    private F first; //first member of pair
    private S second; //second member of pair
 
    public Pair(F first, S second) {
        this.first = first;
        this.second = second;
    }
    public void setFirst(F first) {
        this.first = first;
    }
    public void setSecond(S second) {
        this.second = second;
    }
    public F getFirst() {
        return first;
    }
    public S getSecond() {
        return second;
    }
    
    public boolean oneMatches( Pair<F,S> P_ ) {
    	if ( P_.getFirst().equals(this.first)) return true;
    	if ( P_.getFirst().equals(this.second)) return true;
    	if ( P_.getSecond().equals(this.first )) return true;
    	if ( P_.getSecond().equals(this.second)) return true;
    	return false;
    }
}
