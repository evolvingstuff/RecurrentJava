package loss;

import matrix.Matrix;

public interface Loss {
	void backward(Matrix actualOutput, Matrix targetOutput) throws Exception;
	double measure(Matrix actualOutput, Matrix targetOutput) throws Exception;
}
