package loss;
import matrix.Matrix;

public class LossSumOfSquares implements Loss {

	@Override
	public void backward(Matrix actualOutput, Matrix targetOutput) throws Exception {
		for (int i = 0; i < targetOutput.w.length; i++) {
			double errDelta = actualOutput.w[i] - targetOutput.w[i];
			actualOutput.dw[i] += errDelta;
		}
	}
	
	@Override
	public double measure(Matrix actualOutput, Matrix targetOutput) throws Exception {
		double sum = 0;
		for (int i = 0; i < targetOutput.w.length; i++) {
			double errDelta = actualOutput.w[i] - targetOutput.w[i];
			sum += 0.5 * errDelta * errDelta;
		}
		return sum;
	}
}
