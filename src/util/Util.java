package util;
import java.util.Collections;
import java.util.List;
import java.util.Random;

import matrix.Matrix;

public class Util {
	
	public static int pickIndexFromRandomVector(Matrix probs, Random r) throws Exception {
		double mass = 1.0;
		for (int i = 0; i < probs.w.length; i++) {
			double prob = probs.w[i] / mass;
			if (r.nextDouble() < prob) {
				return i;
			}
			mass -= probs.w[i];
		}
		throw new Exception("no target index selected");
	}
	
	public static double median(List<Double> vals) {
		Collections.sort(vals);
		int mid = vals.size()/2;
		if (vals.size() % 2 == 1) {
			return vals.get(mid);
		}
		else {
			return (vals.get(mid-1) + vals.get(mid)) / 2;
		}
	}
}
