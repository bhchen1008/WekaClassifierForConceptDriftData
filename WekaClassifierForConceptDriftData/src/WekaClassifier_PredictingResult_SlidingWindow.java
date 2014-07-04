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

public class WekaClassifier_PredictingResult_SlidingWindow {

  protected static void writePredictions(Classifier cls, Instances unlabeled, String prediction_file_name)  {
    try {
      unlabeled.setClassIndex(unlabeled.numAttributes()-1);
      Instances labeled = new Instances(unlabeled);

      for (int i = 0; i < unlabeled.numInstances(); i++) {
        System.out.println("Instance(" + Integer.toString(i) + ") value:" + unlabeled.instance(i).value(1));    //this works for all instances
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
    Instances train = DataSource.read(trainDataPath);
//	Instances train = new Instances(new BufferedReader(new FileReader(args[0])));
    train.setClassIndex(train.numAttributes() - 1);
    //測試Accuracy
    Evaluation eval = null;
    
    
    // 2. testing unlabeled data
    String testPath = args[1];
    Instances testAll = DataSource.read(testPath);
    
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
    
    //set sliding window size
    int swSize = Integer.parseInt(args[4]);

    // build classifier
    cls.buildClassifier(train);
    //將model output到檔案裏頭
    FileWriter fwriter = null;
    int inputFileLen = trainDataPath.split("/").length;
    String fileName = trainDataPath.split("/")[inputFileLen-1].split("\\.")[0];
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
    
//    Instances unlabeled = new Instances(new BufferedReader(new FileReader(args[1])));
    Instances testlabeled = new Instances(testAll);
    testlabeled.delete();
    Instances testUnlabeled = new Instances(testAll);
    testUnlabeled.delete();
    
    for(int i=0;i<=testAll.numInstances()-swSize;i++){
    	//測試Accuracy用,放在for迴圈裏頭每次重新計算
        eval = new Evaluation(train);

    	if(i==0){
			// initial testing sliding window
			for(int j=0;j<swSize;j++){
				testlabeled.add(testAll.instance(j));
				testUnlabeled.add(testAll.instance(j));
			}	
		}
		
		else{
			testlabeled.delete(0);
			testlabeled.add(testAll.instance(swSize+i-1));
			testUnlabeled.delete(0);
			testUnlabeled.add(testAll.instance(swSize+i-1));
		}
		
		
		testlabeled.setClassIndex(train.numAttributes() - 1);
		testUnlabeled.setClassIndex(train.numAttributes() - 1);
		
		
		eval.evaluateModel(cls, testlabeled);
		
		//get Accuracy
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
				fileName = testPath.split("/")[inputFileLen-1].split("\\.")[0];
				
//				inputFileLen = testPath.split("/").length;
//				String testFile = testPath.split("/")[inputFileLen-1].split("\\.")[0];
//				String[] name = testFile.split("_");
////    		int nameLen = testPath.split("/").length;
//				fileName = "";
//				for(int j=0;j<name.length-2;j++){
//					if(j!=0)
//						fileName+="_";
//					fileName+= name[j];
//				}
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
		
		
		// make sure that data is compatible
		if (!train.equalHeaders(testlabeled))
			throw new IllegalStateException("Training data and testlabeled data are incompatible!");
		if (!train.equalHeaders(testUnlabeled))
			throw new IllegalStateException("Training data and testUnlabeled data are incompatible!");		
		// make sure that class values are missing
		for (int j = 0; j < testUnlabeled.numInstances(); j++)
			testUnlabeled.instance(j).setClassValue(Instance.missingValue());
		
		
//		String predictionPath = outputPath + "/testingDataPredicted/" + ;
		
		String predictionPath = outputPath + "/" + "testingDataPredicted/" + classifier + "/"
								+ fileName + "_time_" + i + ".arff";
		// write predictions
		writePredictions(cls, testUnlabeled, predictionPath);
    }
    
  }
}