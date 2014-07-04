import weka.core.*;
import weka.core.converters.ConverterUtils.*;
import weka.classifiers.*;
import weka.classifiers.trees.*;

import java.io.*;

public class WekaClassifier_PredictingResult {

  protected static void writePredictions(J48 j48, Instances unlabeled, String prediction_file_name)  {
    try {
      unlabeled.setClassIndex(unlabeled.numAttributes()-1);
      Instances labeled = new Instances(unlabeled);

      for (int i = 0; i < unlabeled.numInstances(); i++) {
        System.out.println("Instance(" + Integer.toString(i) + ") value:" + unlabeled.instance(i).value(1));    //this works for all instances
        double clsLabel = j48.classifyInstance(unlabeled.instance(i));                    //this only works up to instance 28 and then fails thereafter
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
//    Instances train = DataSource.read(args[0]);
	Instances train = new Instances(new BufferedReader(new FileReader(args[0])));
    train.setClassIndex(train.numAttributes() - 1);

    // 2. unlabeled data
//    Instances unlabeled = DataSource.read(args[1]);
    Instances unlabeled = new Instances(new BufferedReader(new FileReader(args[1])));
    unlabeled.setClassIndex(train.numAttributes() - 1);
    // make sure that data is compatible
    if (!train.equalHeaders(unlabeled))
      throw new IllegalStateException("Training data and unlabeled data are incompatible!");
    // make sure that class values are missing
    for (int i = 0; i < unlabeled.numInstances(); i++)
      unlabeled.instance(i).setClassValue(Instance.missingValue());

    // build classifier
    J48 classifier = new J48();
    classifier.buildClassifier(train);

    // write predictions
    writePredictions(classifier, unlabeled, args[2]);
  }
}