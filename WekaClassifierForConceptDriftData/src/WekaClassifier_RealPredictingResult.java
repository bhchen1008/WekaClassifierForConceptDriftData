import weka.core.*;
import weka.core.converters.ConverterUtils.*;
import weka.classifiers.*;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.rules.JRip;
import weka.classifiers.rules.PART;
import weka.classifiers.trees.*;

import java.io.*;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

public class WekaClassifier_RealPredictingResult {

  protected static void writePredictions(Classifier cls, Instances unlabeled, String prediction_file_name)  {
    try {
      unlabeled.setClassIndex(unlabeled.numAttributes()-1);
      Instances labeled = new Instances(unlabeled);

      for (int i = 0; i < unlabeled.numInstances(); i++) {
        //System.out.println("Instance(" + Integer.toString(i) + ") value:" + unlabeled.instance(i).value(1));    //this works for all instances
        double clsLabel = cls.classifyInstance(unlabeled.instance(i));                    //this only works up to instance 28 and then fails thereafter
        labeled.instance(i).setClassValue(clsLabel);
      }

      // write output
      BufferedWriter writer = new BufferedWriter(new FileWriter(prediction_file_name));
      writer.write(labeled.toString());
      writer.flush();
      writer.close();

    }
    catch(Exception e) {
      e.printStackTrace();
      System.out.println(e.getMessage());
    }
  }

  /**
   * Expects three filenames:
   * <ol>
   *   <li>training set</li>
   *   <li>unlabeled set</li>
   *   <li>output file for predictions</li>
   * </ol>
   */
  public static void main(String[] args) throws Exception {
    // load data
    // 1. training data
	String trainDataPath = args[0];
	System.out.println("trainDataPath:"+trainDataPath);
	Instances train = DataSource.read(trainDataPath);
    
//	Instances train = new Instances(new BufferedReader(new FileReader(args[0])));
    train.setClassIndex(train.numAttributes() - 1);
    
    // 2. testing unlabeled data
    String testPath = args[1];
    Instances testAll = DataSource.read(testPath);
    Instances testUnlabeled = new Instances(testAll);
    testUnlabeled.setClassIndex(train.numAttributes() - 1);
    
    // 3. outputPath
    String outputPath = args[2];
    
    //set classifier
    String classifier = args[3];
    Classifier cls = null;
	// train classifier
	switch(classifier){
		case "weka.classifiers.trees.J48":
			System.out.println("weka.classifiers.trees.J48");
			cls = new J48();
			break;
		case "weka.classifiers.rules.PART":
			cls = new PART();
			break;
		case "weka.classifiers.rules.JRip":
			cls = new JRip();
			break;
		case "weka.classifiers.bayes.NaiveBayes":
			cls = new NaiveBayes();
			break;
	}
    
    // build classifier
    cls.buildClassifier(train);
    //將model output到檔案裏頭
    FileWriter fwriter = null;
    int inputFileLen = testPath.split("/").length;
    String fileName = testPath.split("/")[inputFileLen-1];
        
//    Instances unlabeled = new Instances(new BufferedReader(new FileReader(args[1])));
    
    if (!train.equalHeaders(testUnlabeled))
		throw new IllegalStateException("Training data and testUnlabeled data are incompatible!");		
	// make sure that class values are missing
	for (int j = 0; j < testUnlabeled.numInstances(); j++)
		testUnlabeled.instance(j).setClassValue(Instance.missingValue());
    
    
    String predictionPath = outputPath + "/" + "testingDataPredicted/" + classifier + "/"
			+ fileName;// + ".arff";
    
    writePredictions(cls, testUnlabeled, predictionPath);
    
  }
}