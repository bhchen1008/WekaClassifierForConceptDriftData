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
import java.util.Random;
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

public class WekaClassifier_ConceptDriftData_FixedPercent {
  /** the classifier used internally */
  protected Classifier m_Classifier = null;
  
  /** the filter to use */
  protected Filter m_Filter = null;

  /** the training file */
  protected static String m_concept1File = null;

  /** the training instances */
  protected static Instances m_concept1 = null;
  
  /** the training file */
  protected static String m_concept2File = null;

  /** the training instances */
  protected static Instances m_concept2 = null;

  /** for evaluating the classifier */
  protected Evaluation m_Evaluation = null;

  /**
   * initializes the demo
   */
  public WekaClassifier_ConceptDriftData_FixedPercent() {
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
    m_concept1File = name;
    m_concept1     = new Instances(
                        new BufferedReader(new FileReader(m_concept1File)));
    m_concept1.setClassIndex(m_concept1.numAttributes() - 1);
         
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
    m_Evaluation = new Evaluation(m_concept1/*filtered*/);
    m_Evaluation.crossValidateModel(
        m_Classifier, m_concept1/*filtered*/, 10, m_concept1.getRandomNumberGenerator(randomSeed));
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
        + m_concept1File + "\n");
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
        "\nusage:\n  " + WekaClassifier_ConceptDriftData_FixedPercent.class.getName() 
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
    WekaClassifier_ConceptDriftData_FixedPercent         demo;

    if (args.length < 4) {
      System.out.println(WekaClassifier_ConceptDriftData_FixedPercent.usage());
      System.exit(1);
    }
    
    String concept1DataPath = args[0];
    String concept2DataPath = args[1];
    String outputPath = args[2];
    String classifier = args[3];
//    String classifier = "J48";
    
    m_concept1File = concept1DataPath;
	m_concept1     = new Instances(
			new BufferedReader(new FileReader(m_concept1File)));
	
	m_concept1.setClassIndex(m_concept1.numAttributes() - 1);    		    	
	Instances train = new Instances(m_concept1);

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
//Classifier cls = new J48();
//Classifier cls = new NaiveBayes();
	cls.buildClassifier(train);
	// evaluate classifier and print some statistics
	Evaluation eval_Cumulative = new Evaluation(train);//放在外面會變成累積的Accuracy
	
	System.out.println("cls:\n"+cls);
	//將model output到檔案裏頭
	FileWriter fwriter = null;
	int inputFileLen = concept1DataPath.split("/").length;
	String fileName = concept1DataPath.split("/")[inputFileLen-1].split("\\.")[0];
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
    
    for(int time=1;time<=20;time++){
    	int threshold = time*5;
    	for(int count=1;count<=10;count++){   
    		Evaluation eval = new Evaluation(train);//放在裡面每次都是獨立的Accuracy
    		
    		Random rand = new Random();
    		m_concept2File = concept2DataPath;
    		m_concept2     = new Instances(
    				new BufferedReader(new FileReader(m_concept2File)));
    		
    		m_concept2.setClassIndex(m_concept2.numAttributes() - 1);
    		Instances test = new Instances(m_concept2);
    		double randDouble;//random來看要拿哪個pool的data
    		int randInt;//random來看要拿pool裏頭的哪個data
//	m_concept2.numInstances()
//	Instances test = new Instances();
    		test.delete();
    		Instances concept1_pool = new Instances(m_concept1);
    		Instances concept2_pool = new Instances(m_concept2);
//    		if(threshold==10)
//    			System.out.println("here");
    		//此段用來記錄testingData是拿取哪個pool的資料
    		fwriter = null;
    		inputFileLen = concept1DataPath.split("/").length;
    		fileName = concept1DataPath.split("/")[inputFileLen-1].split("\\.")[0];
    		filePath = outputPath + "/" + "testingData/Detail/"
    				+ fileName + "_testingData_time_" + time/10 + "" + time%10 + "_n_" + count + ".csv";
    		saveFile = new File(filePath);
    		fout = new FileOutputStream(saveFile, true);
    		bw = new BufferedWriter(new OutputStreamWriter(fout));
    		bw.flush();
    		int total = 100;//總共要一百筆資料
    		for(int i=1;i<=total;i++){
    			if(i<=100*(double)threshold/100){
    				randInt = rand.nextInt(concept2_pool.numInstances());
    				test.add(concept2_pool.instance(randInt));
    				bw.write("pool,2\n");
//    				concept2_pool.delete(randInt);
    			}
    			else{
    				randInt = rand.nextInt(concept1_pool.numInstances());
    				test.add(concept1_pool.instance(randInt));
    				bw.write("pool,1\n");
//    				concept1_pool.delete(randInt);
    			}
//    			randDouble = rand.nextDouble();
//    			if(randDouble<(double)threshold/10){
//    				randInt = rand.nextInt(concept2_pool.numInstances());
//    				test.add(concept2_pool.instance(randInt));
//    				bw.write("pool,2\n");
//    				concept2_pool.delete(randInt);
//    			}
//    			else{
//    				randInt = rand.nextInt(concept1_pool.numInstances());
//    				test.add(concept1_pool.instance(randInt));
//    				bw.write("pool,1\n");
//    				concept1_pool.delete(randInt);
//    			}
    		}
    		bw.flush();
    		bw.close();
    		fout.close();
    		System.out.println("test:\n"+test);
    		
    		fwriter = null;
    		inputFileLen = concept1DataPath.split("/").length;
    		fileName = concept1DataPath.split("/")[inputFileLen-1].split("\\.")[0];
    		filePath = outputPath + "/" + "testingData/"
    				+ fileName + "_testingData_time_" + time/10 + "" + time%10 + "_n_" + count + ".arff";
    		saveFile = new File(filePath);
    		fout = new FileOutputStream(saveFile, true);
    		bw = new BufferedWriter(new OutputStreamWriter(fout));
    		bw.flush();
    		bw.write(test.toString());
    		bw.flush();
    		bw.close();
    		fout.close();
    		
    		
    		
    		
    		
    		
    		
    		eval.evaluateModel(cls, test);
    		System.out.println(eval.toSummaryString("\nnew Results\n======\n", false));
    		System.out.println(eval.toClassDetailsString());
    		eval_Cumulative.evaluateModel(cls, test);
    		System.out.println(eval_Cumulative.toSummaryString("\nCumulative Results\n======\n", false));
    		System.out.println(eval_Cumulative.toClassDetailsString());
    		
    		
    		String accuracyOutputPath = concept2DataPath;
    		
    		//獨立的Accuracy
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
    				inputFileLen = accuracyOutputPath.split("/").length;
    				fileName = accuracyOutputPath.split("/")[inputFileLen-1].split("\\.")[0];
    				filePath = outputPath + "/" + "AccuracySet/"
    						+ fileName + "_threshold_" + time/10 + "" + time%10 + "_All_Accuracy.csv";
    				
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
    					String fileWriteContent = Accuracy/100 + "";
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
    					fwriter.write(Accuracy/100 + "");
    				}    				    				
    			}
    			if(count!=10){
    				fwriter.write(",");
    			}
    			fwriter.close();    			
    		}
    		
    		//累積的Accuracy
    		String result_Cumulative = eval_Cumulative.toSummaryString("\nResults\n======\n", false);
    		
    		patternStr = "Correctly Classified Instances.*%";
    		pattern = Pattern.compile(patternStr);
    		matcher = pattern.matcher(result_Cumulative);
    		matchFound = matcher.find();
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
    				inputFileLen = accuracyOutputPath.split("/").length;
    				fileName = accuracyOutputPath.split("/")[inputFileLen-1].split("\\.")[0];
    				filePath = outputPath + "/" + "AccuracySet_Cumulative/"
    						+ fileName + "_threshold_" + time/10 + "" + time%10 + "_All_Accuracy.csv";
    				
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
    					String fileWriteContent = Accuracy/100 + "";
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
    					fwriter.write(Accuracy/100 + "");
    				}    				    				
    			}
    			if(count!=10){
    				fwriter.write(",");
    			}
    			fwriter.close();    			
    		}
    		    		    	
    	}
//    	fwriter.write("\n");
//    	if(threshold==2)
//    		threshold=8;
    }
  }
}
