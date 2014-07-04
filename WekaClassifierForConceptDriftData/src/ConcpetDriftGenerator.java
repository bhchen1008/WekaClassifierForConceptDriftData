import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.rules.JRip;
import weka.classifiers.rules.PART;
import weka.classifiers.trees.J48;
import weka.core.Instance;
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
import java.util.ArrayList;
import java.util.HashMap;
import java.util.LinkedList;
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

public class ConcpetDriftGenerator {
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
  public ConcpetDriftGenerator() {
    super();
  }

    /**
   * returns the usage of the class
   */
  public static String usage() {
    return
        "\nusage:\n  " + ConcpetDriftGenerator.class.getName() 
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
  
  public static void main(String[] args) throws Exception {
    ConcpetDriftGenerator         demo;

    if (args.length < 4) {
      System.out.println(ConcpetDriftGenerator.usage());
      System.exit(1);
    }
    String concept1DataPath = "";
    String concept2DataPath = "";
    String outputPath = "";
    String inputDataPath = "";
    int amountOfDF = 0;//Drift Feature的數量
    int dataSize = 0;
    int bigDrift = 0;//是否產生bigDrift
    int bigDriftTime = 0;
    int bigDriftTimeTo = 0;
    int randomFeatureDrift = 0;
    String featureDrift = "";
    
    int conDriftType = Integer.parseInt(args[0]);
    int fixedPercent = Integer.parseInt(args[1]);
    int delDuplicate = Integer.parseInt(args[2]);
    if(conDriftType==0){//是用兩種concept的Data組合而成
    	concept1DataPath = args[3];//concept1DataPath，此種DriftType會從concept1慢慢Drift到concept2
        concept2DataPath = args[4];//concept2DataPath
        outputPath = args[5];//所有結果的outputPath
        dataSize = Integer.parseInt(args[6]);//每個時間點的dataSetSize
    }
    
    else{//1的話則是用一種，然後改變其attribute value
    	inputDataPath = args[3];//inputDataPath，此種DriftType會利用這個Data將其attribute改變(consistent)，來模擬ConceptDrift
    	outputPath = args[4];//所有結果的outputPath
    	amountOfDF = Integer.parseInt(args[5]);//有幾個attribute會Drift
    	dataSize = Integer.parseInt(args[6]);//每個時間點的dataSetSize
    	randomFeatureDrift = Integer.parseInt(args[10]);
    	featureDrift = args[11];
    }
    
    bigDrift = Integer.parseInt(args[7]);//是否發生大幅度的Drift
    if(bigDrift==1){
    	bigDriftTime = Integer.parseInt(args[8]);//大幅度Drift的開始時間
        bigDriftTimeTo = Integer.parseInt(args[9]);//大幅度Drift到哪個時間
    }
    
    
    if(conDriftType==0){
    	m_concept1File = concept1DataPath;
    	m_concept1     = new Instances(
    			new BufferedReader(new FileReader(m_concept1File)));
    	
    	m_concept1.setClassIndex(m_concept1.numAttributes() - 1);    		    	
    	Instances train = new Instances(m_concept1);
    	
    	System.out.println("Generating concept-drift dataset!");
    	if(conDriftType==0){
    		for(int time=1;time<=20;time++){
    			int threshold;
//        		threshold = time*5;//(40-time)*5;
        		//刻意製造落差用
//        		if(time<=15){
//        			threshold = 8;
//        		}
//        		else{
//        			threshold = 100;
//        		}
        		
        		//recurring
        		if(time<=10){
        			threshold = 8;
        		}
        		else if(time<=15){
        			threshold = 100;
        		}
        		else{
        			threshold = 8;
        		}
        		
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
    				FileWriter fwriter = null;
    				int inputFileLen = concept1DataPath.split("/").length;
    				String fileName1 = concept1DataPath.split("/")[inputFileLen-1].split("\\.")[0];
    				String fileName2 = concept2DataPath.split("/")[inputFileLen-1].split("\\.")[0];
    				String filePath = outputPath + "/" + "testingData/Detail/"
    						+ fileName1 + "To" + fileName2 + "_testingData_time_" + time/10 + "" + time%10 + "_n_" + count + ".csv";
    				File saveFile = new File(filePath);
    				FileOutputStream fout = new FileOutputStream(saveFile, true);
    				BufferedWriter bw = new BufferedWriter(new OutputStreamWriter(fout));
    				bw.flush();
    				
    				
    				for(int i=1;i<=dataSize;i++){	//取一百筆資料當一個Set
    					randDouble = rand.nextDouble();
    					if(fixedPercent == 0){
    						if(randDouble<(double)threshold/100){
//    				System.out.println("i"+i);
//    				System.out.println("m_concept2:"+concept2_pool.numInstances());
    							randInt = rand.nextInt(concept2_pool.numInstances());
    							test.add(concept2_pool.instance(randInt));
    							bw.write("pool,2\n");
    							if(delDuplicate == 1)
    								concept2_pool.delete(randInt);
    						}
    						else{
//    				System.out.println("i"+i);
//    				System.out.println("m_concept1:"+concept1_pool.numInstances());
    							randInt = rand.nextInt(concept1_pool.numInstances());
    							test.add(concept1_pool.instance(randInt));
    							bw.write("pool,1\n");
    							if(delDuplicate == 1)
    								concept1_pool.delete(randInt);
    						}
    					}
    					else{
    						if(i<=dataSize*(double)threshold/100){
    							randInt = rand.nextInt(concept2_pool.numInstances());
    							test.add(concept2_pool.instance(randInt));
    							bw.write("pool,2\n");
    							if(delDuplicate == 1)
    								concept2_pool.delete(randInt);
    						}
    						else{
    							randInt = rand.nextInt(concept1_pool.numInstances());
    							test.add(concept1_pool.instance(randInt));
    							bw.write("pool,1\n");
    							if(delDuplicate == 1)
    								concept1_pool.delete(randInt);
    						}
    					}
    				}
    				bw.flush();
    				bw.close();
    				fout.close();
    				
    				
    				fwriter = null;
    				inputFileLen = concept1DataPath.split("/").length;
//    		fileName1 = concept1DataPath.split("/")[inputFileLen-1].split("\\.")[0];
    				filePath = outputPath + "/" + "testingData/"
    						+ fileName1 + "To" + fileName2 + "_testingData_time_" + time/10 + "" + time%10 + "_n_" + count + ".arff";
    				saveFile = new File(filePath);
    				fout = new FileOutputStream(saveFile, true);
    				bw = new BufferedWriter(new OutputStreamWriter(fout));
    				bw.flush();
    				bw.write(test.toString());
    				bw.flush();
    				bw.close();
    				fout.close();
    				
    			}
    			if(bigDrift==1){
    				if(time==bigDriftTime){
    					time = bigDriftTimeTo;
    				}
    			}
//    	fwriter.write("\n");
//    	if(threshold==2)
//    		threshold=8;
    		}
    	}
    	
    }
	//下一種Type of concept-drift
    else{
    	m_concept1File = inputDataPath;
    	m_concept1     = new Instances(
    			new BufferedReader(new FileReader(m_concept1File)));
    	
    	m_concept1.setClassIndex(m_concept1.numAttributes() - 1);    		    	
    	Instances data = new Instances(m_concept1);
    	
    	System.out.println("Generating concept-drift dataset!");
    	
    	Random rand = new Random();
    	ArrayList<Integer> driftList = new ArrayList<Integer>();
    	if(randomFeatureDrift == 1){
    		//決定哪些attribute要drift
    		ArrayList<Integer> intList = new ArrayList<Integer>();
    		for(int i=0;i<data.numAttributes()-1;i++){
    			intList.add(i);
    		}
    		for(int i=0;i<amountOfDF;i++){//開始將attribute drift
    			int randInt = rand.nextInt(intList.size());
    			driftList.add(intList.get(randInt));
    			intList.remove(randInt);
    		}
    	}
    	else{
    		for(String x:featureDrift.split(",")){
    			driftList.add(Integer.parseInt(x));
    		}
    	}
    	
    	//印出選出那些Feature來Drift
    	System.out.print("Drift Feature:");
    	for(int i=0;i<driftList.size();i++){
    		if(i!=0)
    			System.out.print(",");
    		System.out.print(driftList.get(i));
    	}
    	System.out.println();
    	
    	for(int time=1;time<=20;time++){
//    	for(int time=21;time<=40;time++){
    		int threshold;
//    		threshold = time*5;
    		//刻意製造落差用
//    		if(time<=15){
//    			threshold = 0;
//    		}
//    		else{
//    			threshold = 100;
//    		}
    		//recurring
    		if(time<=10){
    			threshold = 0;
    		}
    		else if(time<=15){
    			threshold = 100;
    		}
    		else{
    			threshold = 0;
    		}
    		
    		
//    		int threshold = (40-time)*5;
    		for(int count=1;count<=10;count++){
    			System.out.println("File" + ((time-1)*10+count) + ":");
    			Instances test = new Instances(m_concept1);
    			Instances right = new Instances(m_concept1);//用來存沒有Drift的Data
    			double randDouble;//random來看要拿哪個pool的data
    			int randInt;//random來看要拿pool裏頭的哪個data
    			test.delete();
    			right.delete();
    			Instances concept1_pool = new Instances(m_concept1);
    			for(int i=1;i<=dataSize;i++){	//取一百筆資料當一個Set
    				randDouble = rand.nextDouble();
    				if(fixedPercent == 0){
    					if(randDouble<(double)threshold/100){
        					System.out.println("第 "+i/100+(i/10)%10+i%10+""+" 筆Drift");
        					randInt = rand.nextInt(concept1_pool.numInstances());
        					System.out.println("o:"+concept1_pool.instance(randInt));
        					Instance inst_CD = generateCD(concept1_pool.instance(randInt), driftList);
        					System.out.println("a:"+inst_CD);
        					test.add(inst_CD);
        					right.add(concept1_pool.instance(randInt));
        					if(delDuplicate == 1)
        						concept1_pool.delete(randInt);
        				}
        				else{
        					randInt = rand.nextInt(concept1_pool.numInstances());
        					test.add(concept1_pool.instance(randInt));
        					right.add(concept1_pool.instance(randInt));
        					if(delDuplicate == 1)
        						concept1_pool.delete(randInt);
        				}
    				}
    				else{
    					if(i<=dataSize*(double)threshold/100){
    						System.out.println("第 "+i/100+(i/10)%10+i%10+""+" 筆Drift");
        					randInt = rand.nextInt(concept1_pool.numInstances());
        					System.out.println("o:"+concept1_pool.instance(randInt));
        					Instance inst_CD = generateCD(concept1_pool.instance(randInt), driftList);
        					System.out.println("a:"+inst_CD);
        					test.add(inst_CD);
        					right.add(concept1_pool.instance(randInt));
        					if(delDuplicate == 1)
        						concept1_pool.delete(randInt);
    					}
    					else{
    						randInt = rand.nextInt(concept1_pool.numInstances());
        					test.add(concept1_pool.instance(randInt));
        					right.add(concept1_pool.instance(randInt));
        					if(delDuplicate == 1)
        						concept1_pool.delete(randInt);
    					}
    				}
    				
    			}
    			
    			int inputFileLen = inputDataPath.split("/").length;
    			String fileName1 = inputDataPath.split("/")[inputFileLen-1].split("\\.")[0];
    			FileWriter fwriter = null;
    			String filePath = outputPath + "/" + "testingData/"
    					+ fileName1 + "_testingData_time_" + time/10 + "" + time%10 + "_n_" + count + ".arff";
    			File saveFile = new File(filePath);
    			FileOutputStream fout = new FileOutputStream(saveFile, true);
    			BufferedWriter bw = new BufferedWriter(new OutputStreamWriter(fout));
    			bw.flush();
    			bw.write(test.toString());
    			bw.flush();
    			bw.close();
    			fout.close();
    			
    			fwriter = null;
    			filePath = outputPath + "/" + "testingData/"
    					+ fileName1 + "Right_testingData_time_" + time/10 + "" + time%10 + "_n_" + count + ".arff";
    			saveFile = new File(filePath);
    			fout = new FileOutputStream(saveFile, true);
    			bw = new BufferedWriter(new OutputStreamWriter(fout));
    			bw.flush();
    			bw.write(test.toString());
    			bw.flush();
    			bw.close();
    			fout.close();

    		}
    		if(bigDrift==1){
				if(time==bigDriftTime){
					time = bigDriftTimeTo;
				}
			}
    	}
    }
    
//	else if
    System.out.println("The concept-drift dataset is generated");
  }
  private static Instance generateCD(Instance inst, ArrayList<Integer> driftList){
	  for(int i=0;i<driftList.size();i++){//開始將attribute drift
		  int intAttr = driftList.get(i);
		  int originalPos = inst.attribute(intAttr).indexOfValue(inst.stringValue(intAttr));
		  int newPos = 0;
		  if(originalPos != inst.attribute(intAttr).numValues()-1){//當attribute值為最後一個時，回到第一個
			  newPos = originalPos + 1;
		  }
		  inst.setValue(intAttr, inst.attribute(intAttr).value(newPos));//將值代換掉
	  }
	  return inst;
  }
}
