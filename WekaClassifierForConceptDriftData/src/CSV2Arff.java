import weka.core.Instances;
import weka.core.converters.ArffSaver;
import weka.core.converters.CSVLoader;
 
import java.io.File;
import java.io.FileWriter;
 
public class CSV2Arff {
  /**
   * takes 2 arguments:
   * - CSV input file
   * - ARFF output file
   */
  public static void main(String[] args) throws Exception {
	System.out.println(args.length);
    if (args.length != 2) {
      System.out.println("\nUsage: CSV2Arff <input.csv> > <output.arff>\n");
      System.exit(1);
    }
 
    // load CSV
    CSVLoader loader = new CSVLoader();
    loader.setSource(new File(args[0]));
    Instances data = loader.getDataSet();
    
    String data_s = data.toString();
//    System.out.println(data);
    File saveFile=new File(args[1]);
    try
    {
      FileWriter fwriter=new FileWriter(saveFile);
      fwriter.write(data_s);
      fwriter.close();
    }
    catch(Exception e)
    {
      e.printStackTrace();
    }
 
    // save ARFF
//    ArffSaver saver = new ArffSaver();
//    saver.setInstances(data);
//    saver.setFile(new File(args[1]));
//    saver.setDestination(new File(args[1]));
//    saver.writeBatch();
  }
}