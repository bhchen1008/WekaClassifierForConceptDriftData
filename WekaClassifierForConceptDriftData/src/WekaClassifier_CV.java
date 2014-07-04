import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
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

public class WekaClassifier_CV {
  /** the classifier used internally */
  protected Classifier m_Classifier = null;
  
  /** the filter to use */
  protected Filter m_Filter = null;

  /** the training file */
  protected String m_TrainingFile = null;

  /** the training instances */
  protected Instances m_Training = null;

  /** for evaluating the classifier */
  protected Evaluation m_Evaluation = null;

  /**
   * initializes the demo
   */
  public WekaClassifier_CV() {
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
        "\nusage:\n  " + WekaClassifier_CV.class.getName() 
        + "  CLASSIFIER <classname> [options] \n"
        + "  FILTER <classname> [options]\n"
        + "  DATASET <trainingfile>\n\n"
        + "e.g., \n"
        + "  java -classpath \".:weka.jar\" WekaDemo \n"
        + "    CLASSIFIER weka.classifiers.trees.J48 -U \n"
        + "    FILTER weka.filters.unsupervised.instance.Randomize \n"
        + "    DATASET iris.arff\n";
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
    WekaClassifier_CV         demo;

    if (args.length < 8) {
      System.out.println(WekaClassifier_CV.usage());
      System.exit(1);
    }

    // parse command line
    String classifier = "";
    String filter = "";
    String dataset = "";
    String randomSeed = "";
    String outputPath = "";
    Vector classifierOptions = new Vector();
    Vector filterOptions = new Vector();

    int i = 0;
    String current = "";
    boolean newPart = false;
    do {
      // determine part of command line
      if (args[i].equals("CLASSIFIER")) {
        current = args[i];
        i++;
        newPart = true;
      }
      else if (args[i].equals("FILTER")) {
        current = args[i];
        i++;
        newPart = true;
      }
      else if (args[i].equals("DATASET")) {
        current = args[i];
        i++;
        newPart = true;
      }
      else if (args[i].equals("RANDOM")) {
    	  current = args[i];
    	  i++;
    	  newPart = true;
      }
      else if (args[i].equals("OUTPUT")) {
    	  current = args[i];
    	  i++;
    	  newPart = true;
      }
      
      boolean first_C = true;
      if (current.equals("CLASSIFIER")) {
        if (newPart)
          classifier = args[i];
        else
          classifierOptions.add(args[i]);
      }
      else if (current.equals("FILTER")) {
        if (newPart)
          filter = args[i];
        else
          filterOptions.add(args[i]);
      }
      else if (current.equals("DATASET")) {
        if (newPart)
          dataset = args[i];
      }
      else if (current.equals("RANDOM")) {
    	  if(newPart)
    		  randomSeed = args[i];
      }
      else if (current.equals("OUTPUT")) {
    	  if(newPart)
    		  outputPath = args[i];
      }

      // next parameter
      i++;
      newPart = false;
    } 
    while (i < args.length);

    // everything provided?
//    if ( classifier.equals("") || filter.equals("") || dataset.equals("") ) {
    if ( classifier.equals("") || dataset.equals("") || randomSeed.equals("") || outputPath.equals("") ) {
      System.out.println("Not all parameters provided!");
      System.out.println(WekaClassifier_CV.usage());
      System.exit(2);
    }

    // run
    demo = new WekaClassifier_CV();
    demo.setClassifier(
        classifier, 
        (String[]) classifierOptions.toArray(new String[classifierOptions.size()]));
//    demo.setFilter(
//        filter,
//        (String[]) filterOptions.toArray(new String[filterOptions.size()]));
    demo.setTraining(dataset);
    demo.execute(Integer.parseInt(randomSeed));
    System.out.println(demo.toString());
    String result = demo.toString();
    
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
    		FileWriter fwriter = null;
    		int inputFileLen = dataset.split("/").length;
    		String fileName = dataset.split("/")[inputFileLen-1].split("\\.")[0];
    		String filePath = outputPath + "/" + "AccuracySet/"
					+ fileName + "_r_" + randomSeed + "_All_Accuracy.txt";
    		 
    		File saveFile = new File(filePath);
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
    			String fileWriteContent = classifier + ":\n" + "Accuracy:" + Accuracy/100 + "\n\n";
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
    			fwriter.write(classifier + ":\n" + "Accuracy:" + Accuracy/100 + "\n\n");
    		}
    		fwriter.close();
    		
    		//Collect all accuracy to CSVFile 後來用Python解決這部分
//    		if(fileName.split("_").length > 1)
//    			fileName = fileName.split("_p_")[0] + "_n" + fileName.split("_p_")[1].split("_n")[1];
//    		saveFile = new File(outputPath + "/" + "AccuracySet/"
//					+ fileName + "_AccuracySet.csv");
//    		if(saveFile.exists()){	//若檔案存在
//    			freader= new FileReader(outputPath + "/" + "AccuracySet/"
//						+ fileName + "_AccuracySet.csv");
//    			num = freader.read(data);
//    			if(num>0)
//    				str = new String(data,0,num);    		
//    			
//    			//寫檔
//    			fwriter = new FileWriter(saveFile);
//    			String Accuracy_s = matcher.group(0).toString();
//    			String Accuracy_s1 = Accuracy_s.split("%")[0];
//    			double Accuracy = Double.parseDouble(Accuracy_s1);
//    			if(num>0)
//    				fwriter.write(str + Accuracy/100 + ",");
//    			else
//    				fwriter.write(Accuracy/100 + ",");
//    		}
//    		else{
//    			fwriter = new FileWriter(saveFile);
//    			String Accuracy_s = matcher.group(0).toString();
//    			String Accuracy_s1 = Accuracy_s.split("%")[0];
//    			double Accuracy = Double.parseDouble(Accuracy_s1);
//    			fwriter.write(Accuracy/100 + ",");
//    		}
//    		fwriter.close();
    		
    		//根據algorithm名稱的writer 依然用Python來將所有Accuracy整合即可
//    		int algoLen = classifier.toString().split("\\.").length;
//    		String fileName_algo = classifier.split("\\.")[algoLen-1];
//    		if (dataset.split("/")[inputFileLen-1].split("\\.")[0].split("_p").length > 1)
//    			fileName = dataset.split("/")[inputFileLen-1].split("\\.")[0].split("_p")[0] + "_n" +
//    						dataset.split("/")[inputFileLen-1].split("\\.")[0].split("_p")[1].split("_n")[1];
//    		else
//    			fileName = dataset.split("/")[inputFileLen-1].split("\\.")[0];
//    		String dirName = fileName.split("_")[0]; 
//    		filePath = outputPath + "/" + dirName + "/"
//					+ fileName_algo + "_" + fileName + 
//					"_All_Accuracy.csv";
//    		saveFile = new File(filePath);
//    		if(saveFile.exists()){	//若檔案存在
//        		freader= new FileReader(filePath);
//        		num = freader.read(data);
//        		if(num>0)
//        			str = new String(data,0,num);
//        		
//        		//寫檔
//    			fwriter = new FileWriter(saveFile);
//    			String Accuracy_s = matcher.group(0).toString();
//    			String Accuracy_s1 = Accuracy_s.split("%")[0];
//    			double Accuracy = Double.parseDouble(Accuracy_s1);
//    			if(num>0)
//    				fwriter.write(str + Accuracy/100 + ",");
//    			else
//    				fwriter.write(Accuracy/100 + ",");
//    		}
//    		else{
//    			fwriter = new FileWriter(saveFile);
//    			String Accuracy_s = matcher.group(0).toString();
//    			String Accuracy_s1 = Accuracy_s.split("%")[0];
//    			double Accuracy = Double.parseDouble(Accuracy_s1);
//    			fwriter.write(Accuracy/100 + ",");
//    		}
//    		fwriter.close();
    	}
    }      
  }
}
