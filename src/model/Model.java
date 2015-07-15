package model;
import java.io.Serializable;
import java.util.List;

import matrix.Matrix;
import autodiff.Graph;


public interface Model extends Serializable {
	Matrix forward(Matrix input, Graph g) throws Exception;
	void resetState();
	List<Matrix> getParameters();
}
