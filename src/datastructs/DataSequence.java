package datastructs;
import java.util.ArrayList;
import java.util.List;


public class DataSequence {
	public List<DataStep> steps = new ArrayList<>();
	
	public DataSequence() {
		
	}
	
	public DataSequence(List<DataStep> steps) {
		this.steps = steps;
	}
	
	@Override
	public String toString() {
		String result = "";
		result += "========================================================\n";
		for (DataStep step : steps) {
			result += step.toString() + "\n";
		}
		result += "========================================================\n";
		return result;
	}
}
