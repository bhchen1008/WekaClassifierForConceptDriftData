import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.rules.JRip;
import weka.classifiers.rules.PART;
import weka.classifiers.trees.J48;
import weka.core.Instances;
import weka.core.OptionHandler;
import weka.core.Utils;
import weka.filters.Filter;

import java.io.File;
import java.io.FileReader;
import java.io.BufferedReader;
import java.io.FileWriter;
import java.util.Vector;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * A little demo java program for using WEKA.<br/>
 * Check out the Evaluation class for more details.
 *
 * @author     FracPete (fracpete at waikato dot ac dot nz)
 * @see        Evaluation
 */

public class DataGenerator_TrainTest {
  /** the original file */
  protected static String m_OriginalFile = null;

  /** the original instances */
  protected static Instances m_Original = null;
  
  /** the training file */
  protected static String m_TrainingFile = null;

  /** the training instances */
  protected static Instances m_Training = null;
  
  /** the testing file */
  protected static String m_TestingFile = null;

  /** the testing instances */
  protected static Instances m_Testing = null;

  /** for evaluating the classifier */
  protected Evaluation m_Evaluation = null;

  /**
   * initializes the demo
   */
  public DataGenerator_TrainTest() {
    super();
  }


  /**
   * runs 10fold CV over the training file
   */
  public void execute(int randomSeed) throws Exception {
    // run filter
//    m_Filter.setInputFormat(m_Training);
//    Instances filtered = Filter.useFilter(m_Training, m_Filter);
    
    // train classifier on complete file for tree
//    m_Classifier.buildClassifier(filtered);
    
    // 10fold CV with seed=1
//    m_Evaluation = new Evaluation(m_Original/*filtered*/);
//    m_Evaluation.crossValidateModel(
//        m_Classifier, m_Original/*filtered*/, 10, m_Original.getRandomNumberGenerator(randomSeed));
  }

  /**
   * returns the usage of the class
   */
  public static String usage() {
    return
        "\nusage:\n  " + DataGenerator_TrainTest.class.getName() 
        + "  OriginalDataPath \n"
        + "  TrainingDataOutputPath\n"
        + "  TestingDataOutputPath\n";
  }
  
  /**
   * runs the program, the command line looks like this:<br/>
   * WekaDemo CLASSIFIER classname [options] 
   *          FILTER classname [options] 
   *          DATASET filename 
   * <br/>
   * e.g., <br/>
   *   java -classpath ".:weka.jar" WekaDemo \<br/>
   *     CLASSIFIER weka.classifiers.trees.J48 -U \<br/>
   *     FILTER weka.filters.unsupervised.instance.Randomize \<br/>
   *     DATASET iris.arff<br/>
   */
  public static void main(String[] args) throws Exception {
    DataGenerator_TrainTest         demo;

    if (args.length < 3) {
      System.out.println(DataGenerator_TrainTest.usage());
      System.exit(1);
    }
    FileWriter fwriter = null;
    
    String originalPath = args[0];
    String trainOutputPath = args[1];
    String testOutputPath = args[2];
    
    m_OriginalFile = originalPath;
	m_Original     = new Instances(
            new BufferedReader(new FileReader(m_OriginalFile)));		
	    		
	m_Original.setClassIndex(m_Original.numAttributes() - 1);    		    	
	Instances original = new Instances(m_Original);
	
	Instances trainingData = original.trainCV(10, 1);
	Instances testingData = original.testCV(10, 1);
	
	File trainFile = new File(trainOutputPath);
	File testFile = new File(testOutputPath);
	
	if(trainFile.exists()){
		System.out.println(trainOutputPath+"is exist.");
	}
	fwriter = new FileWriter(trainFile);
	fwriter.write(trainingData.toString());
	
	if(trainFile.exists()){
		System.out.println(testOutputPath+"is exist.");
	}
	fwriter = new FileWriter(testFile);
	fwriter.write(testingData.toString());
	
  }
}
