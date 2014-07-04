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

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileOutputStream;
import java.io.FileReader;
import java.io.BufferedReader;
import java.io.FileWriter;
import java.io.OutputStreamWriter;
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

public class WekaClassifier_TrainSlidingWindow {
  /** the classifier used internally */
  protected Classifier m_Classifier = null;
  
  /** the filter to use */
  protected Filter m_Filter = null;

  /** the training file */
  protected static String m_TrainingFile = null;

  /** the training instances */
  protected static Instances m_Training = null;
  
  /** the training file */
  protected static String m_TestingFile = null;

  /** the training instances */
  protected static Instances m_Testing = null;

  /** for evaluating the classifier */
  protected Evaluation m_Evaluation = null;

  /**
   * initializes the demo
   */
  public WekaClassifier_TrainSlidingWindow() {
    super();
  }

  /**
   * sets the classifier to use
   * @param name        the classname of the classifier
   * @param options     the options for the classifier
   */
  public void setClassifier(String name, String[] options) throws Exception {
    m_Classifier = Classifier.forName(name, options);
  }

  /**
   * sets the filter to use
   * @param name        the classname of the filter
   * @param options     the options for the filter
   */
  public void setFilter(String name, String[] options) throws Exception {
    m_Filter = (Filter) Class.forName(name).newInstance();
    if (m_Filter instanceof OptionHandler)
      ((OptionHandler) m_Filter).setOptions(options);
  }

  /**
   * sets the file to use for training
   */
  public void setTraining(String name) throws Exception {
    m_TrainingFile = name;
    m_Training     = new Instances(
                        new BufferedReader(new FileReader(m_TrainingFile)));
    m_Training.setClassIndex(m_Training.numAttributes() - 1);
         
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
    m_Evaluation = new Evaluation(m_Training/*filtered*/);
    m_Evaluation.crossValidateModel(
        m_Classifier, m_Training/*filtered*/, 10, m_Training.getRandomNumberGenerator(randomSeed));
  }

  /**
   * outputs some data about the classifier
   */
  public String toString() {
    StringBuffer        result;

    result = new StringBuffer();
    result.append("Weka - Demo\n===========\n\n");

    result.append("Classifier...: " 
        + m_Classifier.getClass().getName() + " " 
        + Utils.joinOptions(m_Classifier.getOptions()) + "\n");
    if (m_Filter instanceof OptionHandler)
      result.append("Filter.......: " 
          + m_Filter.getClass().getName() + " " 
          + Utils.joinOptions(((OptionHandler) m_Filter).getOptions()) + "\n");
    else
//      result.append("Filter.......: "
//          + m_Filter.getClass().getName() + "\n");
    result.append("Training file: " 
        + m_TrainingFile + "\n");
    result.append("\n");

    result.append(m_Classifier.toString() + "\n");
    result.append(m_Evaluation.toSummaryString() + "\n");
    try {
      result.append(m_Evaluation.toMatrixString() + "\n");
    }
    catch (Exception e) {
      e.printStackTrace();
    }
    try {
      result.append(m_Evaluation.toClassDetailsString() + "\n");
    }
    catch (Exception e) {
      e.printStackTrace();
    }
    
    return result.toString();
  }

  /**
   * returns the usage of the class
   */
  public static String usage() {
    return
        "\nusage:\n  " + WekaClassifier_TrainSlidingWindow.class.getName() 
        + "  <trainingfile>\n"
        + "  <testingfile>\n"
        + "  <Output_Result_Path>\n"
        + "  <CLASSIFIER_name>\n\n"
        + "e.g., \n"
        + "  java -jar WekaClassifier_TrainTest.jar \n"
        + "    Training Dataset \n"
        + "    Testing Dataset\n"
        + "    Output_Result_Path\n"
        + "    CLASSIFIER weka.classifiers.trees.J48 \n";
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
    WekaClassifier_TrainSlidingWindow         demo;

    if (args.length < 4) {
      System.out.println(WekaClassifier_TrainSlidingWindow.usage());
      System.exit(1);
    }
    
    String dataPath = args[0];
    String testPath = args[1];
    String outputPath = args[2];
    String classifier = args[3];
    int swSize = Integer.parseInt(args[4]);//sliding window的size
//    String classifier = "J48";
    
    m_TrainingFile = dataPath;
	m_Training     = new Instances(
            new BufferedReader(new FileReader(m_TrainingFile)));
	    		
	m_Training.setClassIndex(m_Training.numAttributes() - 1);    		    	
	Instances train = new Instances(m_Training);
	
	m_TestingFile = testPath;
	m_Testing     = new Instances(
            new BufferedReader(new FileReader(m_TestingFile)));
	    		
	m_Testing.setClassIndex(m_Testing.numAttributes() - 1);
	
	Instances testAll = new Instances(m_Testing);//整份testing data
	Instances test = new Instances(m_Testing);
	test.delete();
	
	
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
//	Classifier cls = new J48();
//	Classifier cls = new NaiveBayes();
	cls.buildClassifier(train);
	// evaluate classifier and print some statistics
	Evaluation eval = new Evaluation(train);
//	System.out.println("test:\n"+test);
//	System.out.println("cls:\n"+cls);
	
	//將model output到檔案裏頭
	FileWriter fwriter = null;
	int inputFileLen = dataPath.split("/").length;
	String fileName = dataPath.split("/")[inputFileLen-1].split("\\.")[0];
	String filePath = outputPath + "/" + "model/"
			+ fileName + "_model_" + classifier + ".txt";
	File saveFile = new File(filePath);
	FileOutputStream fout = new FileOutputStream(saveFile, true);
	BufferedWriter bw = new BufferedWriter(new OutputStreamWriter(fout));
	bw.flush();
	bw.write(cls.toString());
	bw.flush();
	bw.close();
	fout.close();
	
	
//	int swId = 0;//sliding window ID
	for(int i=1;i<=testAll.numInstances()-swSize+1;i++){
		if(i==1){
			// initial testing sliding window
			for(int j=0;j<swSize;j++){
				test.add(testAll.instance(j));
			}	
		}
		else{
			test.delete(0);
			test.add(testAll.instance(swSize+i-1));
		}
		
		eval.evaluateModel(cls, test);
		
//	System.out.println(eval.toSummaryString("\nResults\n======\n", false));
//	System.out.println(eval.toClassDetailsString());
		
		
		String result = eval.toSummaryString("\nResults\n======\n", false);
		
		String patternStr = "Correctly Classified Instances.*%";
		Pattern pattern = Pattern.compile(patternStr);
		Matcher matcher = pattern.matcher(result);
		boolean matchFound = matcher.find();
		String accuracyLine;
		if(matchFound) {
			accuracyLine = matcher.group(0);
			patternStr = " {14}.* %";
			pattern = Pattern.compile(patternStr);
			matcher = pattern.matcher(accuracyLine.toString());
			matchFound = matcher.find();
			if(matchFound){
				char data[] = new char[10240];
				FileReader freader = null;
				int num;//檔案長度
				String str = "";//原本檔案裏頭的內容
				//根據data名稱的writer
				fwriter = null;
				inputFileLen = testPath.split("/").length;
				String testFile = testPath.split("/")[inputFileLen-1].split("\\.")[0];
				String[] name = testFile.split("_");
//    		int nameLen = testPath.split("/").length;
				fileName = "";
				for(int j=0;j<name.length-2;j++){
					if(j!=0)
						fileName+="_";
					fileName+= name[j];
				}
//    		fileName = testPath.split("/")[inputFileLen-1].split("\\.")[0];
				filePath = outputPath + "/" + "AccuracySet_"+classifier+"/"
						+ fileName + "_All_Accuracy.csv";
				
				saveFile = new File(filePath);
				if(saveFile.exists()){	//若檔案存在
					freader= new FileReader(filePath);
					num = freader.read(data);
					if(num>0)
						str = new String(data,0,num);    		
					
					//寫檔
					fwriter = new FileWriter(saveFile);
					String Accuracy_s = matcher.group(0).toString();
					String Accuracy_s1 = Accuracy_s.split("%")[0];
					double Accuracy = Double.parseDouble(Accuracy_s1);
					String fileWriteContent = Accuracy/100 + ",";
					if(num>0)
						fwriter.write(str + fileWriteContent);
					else
						fwriter.write(fileWriteContent);
				}
				else{
					fwriter = new FileWriter(saveFile);
					String Accuracy_s = matcher.group(0).toString();
					String Accuracy_s1 = Accuracy_s.split("%")[0];
					double Accuracy = Double.parseDouble(Accuracy_s1);
					fwriter.write( Accuracy/100 + ",");
				}
				fwriter.close();
				
			}
		}      
		
	}	
	
  }
}
