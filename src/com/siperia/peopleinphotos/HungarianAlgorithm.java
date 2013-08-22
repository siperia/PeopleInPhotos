package com.siperia.peopleinphotos;

import java.util.ArrayList;
import java.util.List;

// Hungarian Algorithm for selecting best combination of values to minimize the total cost
// Mostly from wikipedia article: http://en.wikipedia.org/wiki/Hungarian_algorithm#Matrix_interpretation
// http://stackoverflow.com/questions/14795111/hungarian-algorithm-how-to-cover-0-elements-with-minimum-lines?rq=1
public class HungarianAlgorithm {
		
	double[][] data;
	boolean [][] mask;
	
	int first_dummy;
	
	enum LineType { NONE, HORIZONTAL, VERTICAL }
	private static class Line {
        int lineIndex;
        LineType rowType;
        Line(int lineIndex, LineType rowType) { 
            this.lineIndex = lineIndex;
            this.rowType = rowType;
        }      
        LineType getLineType() {
            return rowType;
        }

        int getLineIndex() { 
            return lineIndex; 
        }
        boolean isHorizontal() {
            return rowType == LineType.HORIZONTAL;
        }
    }
	
	public HungarianAlgorithm( double[][] in, int dummy ) {
		data = in;
		first_dummy = dummy;
		
		startingReduction();
				
		for(;;) {
			List<Line> minLines = getMinLines();
			if (minLines.size() == data.length) break; // done
			
			double minUncovered = minUncoveredElement( minLines );
			
			for (Line line: minLines) {
				if (line.isHorizontal()) {
					addToRow( line.getLineIndex(), minUncovered);
				} else {
					addToCol( line.getLineIndex(), minUncovered);
				}
			}
			
			subtractTotalMin();
		}
	}
	
	/*public int[] selection() {
		int[] retval = new int[data.length];
	
		for (int i=0;i<data.length;i++) {
	}*/
	
	private void subtractTotalMin() {
		double min = Double.MAX_VALUE;
		double[] rowmins = minOnRows();
		double[] colmins = minOnCols();
		
		for (int i = 0; i < data.length; i++) {
			if (rowmins[i] < min) min = rowmins[i];
			if (colmins[i] < min) min = colmins[i];
		}
		
		for (int i = 0; i < data.length; i++) {
			addToRow(i, -min);
		}
	}
	
	private void startingReduction() {
		double[] mins = minOnRows();
		for (int i=0;i<data.length;i++) {
			for (int j=0;j<data[i].length;j++) data[i][j] -= mins[i];
		}
		mins = minOnCols();
		for (int i=0;i<data.length;i++) {
			for (int j=0;j<data[i].length;j++) data[j][i] -= mins[i];
		}
	}
	
	private double minUncoveredElement(List<Line> lines) {
		mask = new boolean[data.length][data.length]; // to zero it
		
		for (Line line: lines) {
			if (line.isHorizontal()) {
				for (int i=0;i<data.length;i++) mask[i][line.getLineIndex()] = true;				
			} else {
				for (int j=0;j<data.length;j++) mask[line.getLineIndex()][j] = true;
			}
		}
		
		double min = Double.MAX_VALUE;
		for (int i=0;i<data.length;i++) {
			for (int j=0;j<data.length;j++) {
				if (!mask[i][j] && (data[i][j] < min) ) min = data[i][j];
			}
		}
		
		mask = null;
		return min;
	}
	
	private void addToRow(int i, double val) {
		for (int j=0;j<data.length;j++) data[i][j] += val;
	}
	
	private void addToCol(int j, double val) {
		for (int i=0;i<data.length;i++) data[i][j] += val;
	}
	
	private double[] minOnRows() {
		double[] mins = new double[data.length];
		for (int i = 0;i < data.length; i++) {
			double min = Double.MAX_VALUE;		
			for (int j = 0;j < data.length; j++) {
				if ( data[i][j] < min ) min = data[i][j];
			}
			mins[i] = min;
		}
		return mins;
	}
	
	private double[] minOnCols() {
		double[] mins = new double[data.length];
		for (int j = 0;j < data.length; j++) {
			double min = Double.MAX_VALUE;
			for (int i = 0;i < data.length; i++) {
				if ( data[i][j] < min ) min = data[i][j];
			}
			mins[j] = min;
		}
		return mins;
	}
	
	
	private static boolean isZero(int[] array) {
        for (int e : array) {
            if (e != 0) {
                return false;
            }
        }
        return true;
    }
	
	public List<Line> getMinLines() {        
        final int SIZE = data.length;
        int[] zerosPerRow = new int[SIZE];
        int[] zerosPerCol = new int[SIZE];

        // Count the number of 0's per row and the number of 0's per column        
        for (int i = 0; i < SIZE; i++) { 
            for (int j = 0; j < SIZE; j++) { 
                if (data[i][j] == 0) { 
                    zerosPerRow[i]++;
                    zerosPerCol[j]++;
                }
            }
        }

        // There should be at must SIZE lines, initialize the list with an initial capacity of SIZE
        List<Line> lines = new ArrayList<Line>(SIZE);

        LineType lastInsertedLineType = LineType.NONE;

        // While there are 0's to count in either rows or colums...
        while (!isZero(zerosPerRow) && !isZero(zerosPerCol)) { 
            // Search the largest count of 0's in both arrays
            int max = -1;
            Line lineWithMostZeros = null;
            for (int i = 0; i < SIZE; i++) {
                // If exists another count of 0's equal to "max" but in this one has
                // the same direction as the last added line, then replace it with this
                // 
                // The heuristic "fixes" the problem reported by @JustinWyss-Gallifent and @hkrish
                if (zerosPerRow[i] > max || (zerosPerRow[i] == max && lastInsertedLineType == LineType.HORIZONTAL)) {
                    lineWithMostZeros = new Line(i, LineType.HORIZONTAL);
                    max = zerosPerRow[i];
                }
            }

            for (int i = 0; i < SIZE; i++) {
                // Same as above
                if (zerosPerCol[i] > max || (zerosPerCol[i] == max && lastInsertedLineType == LineType.VERTICAL)) {
                    lineWithMostZeros = new Line(i, LineType.VERTICAL);
                    max = zerosPerCol[i];
                }
            }

            // Delete the 0 count from the line 
            if (lineWithMostZeros.isHorizontal()) {
                zerosPerRow[lineWithMostZeros.getLineIndex()] = 0; 
            } else {
                zerosPerCol[lineWithMostZeros.getLineIndex()] = 0;
            }

            int index = lineWithMostZeros.getLineIndex(); 
            if (lineWithMostZeros.isHorizontal()) {
                for (int j = 0; j < SIZE; j++) {
                    if (data[index][j] == 0) {
                        zerosPerCol[j]--;
                    }
                }
            } else {
                for (int j = 0; j < SIZE; j++) {
                    if (data[j][index] == 0) {
                        zerosPerRow[j]--;
                    }
                }                    
            }

            // Add the line to the list of lines
            lines.add(lineWithMostZeros); 
            lastInsertedLineType = lineWithMostZeros.getLineType();
        }
        return lines;
    }   

}
