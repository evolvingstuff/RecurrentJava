import java.util.Random;

import model.Model;
import trainer.Trainer;
import util.NeuralNetworkHelper;
import datasets.TextGeneration;
import datastructs.DataSet;

public class ExamplePaulGraham {
	public static void main(String[] args) throws Exception {
		
		/*
		 * Character-by-character sentence prediction and generation, closely following the example here:
		 * http://cs.stanford.edu/people/karpathy/recurrentjs/
		*/
		
		String textSource = "PaulGraham";
		DataSet data = new TextGeneration("datasets/text/"+textSource+".txt");
		String savePath = "saved_models/"+textSource+".ser";
		boolean initFromSaved = true; //set this to false to start with a fresh model
		boolean overwriteSaved = true;
		
		TextGeneration.reportSequenceLength = 100;
		TextGeneration.singleWordAutocorrect = false; //set this to true to constrain generated sentences to contain only words observed in the training data.

		int bottleneckSize = 10; //one-hot input is squeezed through this
		int hiddenDimension = 200;
		int hiddenLayers = 1;
		double learningRate = 0.001;
		double initParamsStdDev = 0.08;
		
		Random rng = new Random();
		Model lstm = NeuralNetworkHelper.makeLstmWithInputBottleneck( 
				data.inputDimension, bottleneckSize, 
				hiddenDimension, hiddenLayers, 
				data.outputDimension, data.getModelOutputUnitToUse(), 
				initParamsStdDev, rng);
		
		int reportEveryNthEpoch = 10;
		int trainingEpochs = 1000;
		
		Trainer.train(trainingEpochs, learningRate, lstm, data, reportEveryNthEpoch, initFromSaved, overwriteSaved, savePath, rng);
		
		System.out.println("done.");
	}
}
