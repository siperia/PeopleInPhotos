package com.siperia.peopleinphotos;

// Simple value-name pair capable to be sorted as a list
public class Similarity implements Comparable<Similarity> {
    double score;
    String string;

    public Similarity(double score, String string) {
        this.score = score;
        this.string = string;
    }

    @Override
    public int compareTo(Similarity o) {
        return score < o.score ? -1 : score > o.score ? 1 : 0;
    }

}
