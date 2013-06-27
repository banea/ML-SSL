//import java.util.logging.Level;
//import java.util.logging.Logger;
import weka.core.converters.ConverterUtils.DataSource;
import weka.core.Instances;
import weka.classifiers.Classifier;
import weka.core.converters.ArffLoader.ArffReader;
import weka.core.converters.ArffSaver;
import weka.core.Instance;
import weka.classifiers.Evaluation;
import java.io.*;
import java.util.*;

//import java.text.DateFormat;

public class RegressionTesting {
  private static final double DEVIATION = 0.4;
  private static final double subjRegThreshold = 1 - DEVIATION; // threshold has
                                                                // to be in the
                                                                // 0-1.0 range
  private static final double objRegThreshold = DEVIATION; // threshold has to
                                                           // be in the 0-1.0
                                                           // range
  private static final String traindatapath = "/local/carmen/MachineLearning/SenseLevel/ARFF/monolingual/regression/";
  private static final String testdatapath = "/local/carmen/MachineLearning/SenseLevel/ARFF/test/monolingual/regression/";
  private static String FS = "";
  // private static final String FS="fs-PC-"; //feature selection (in case the
  // arff file name starts with some feature selection suffix
  // if nothing, then specify empty string
  // private static String sampleLabel="original"; //exactly the output of the
  // regression algorithm
  private static String sampleLabel = "mapped"; // the output of the regression
                                                // algorithm is mapped to 0,
  // if the label is closer to 0, or to 1, if it is closer to 1.
  private static final int MAXFOLDS = 3;
  // !!!!!modified the MAXITERATIONS number from 10 to 20 to continue from where
  // it left of
  private static final int LASTITERATION = 0;
  private static final String TYPE = "bal";
  private static final String[] LANGUAGES = { "en", "ro" };
  private static final Double[] CLASSES = { 0.0, 1.0 }; // 0 - obj, 1 -subj
  private static BufferedWriter output;
  private static boolean debug = true;

  public static void main(String[] args) throws Exception {

    output = new BufferedWriter(new FileWriter(testdatapath + FS + "results-"
        + DEVIATION + "-" + LASTITERATION + ".txt"));
    for (int fold = 0; fold < MAXFOLDS; fold++) {
      // returns trained classifiers and evaluates classifier on labeled test
      // data
      toLog("entering English trainClassifier function");
      Classifier enClassifier = trainClassifier(LANGUAGES[0], fold,
          LASTITERATION);
      toLog("entering Romanian trainClassifier function");
      Classifier roClassifier = trainClassifier(LANGUAGES[1], fold,
          LASTITERATION);
    }
    output.close();
  }

  private static Classifier trainClassifier(String language, int fold,
      int iteration) {
    // load train data
    toLog("trainClassifier - starts loading data");
    Instances curTrain = loadDataset(traindatapath + FS + TYPE + "-train-"
        + language + "-" + fold + "-" + iteration + ".arff", "train");
    // set class attribute
    curTrain.setClassIndex(curTrain.numAttributes() - 1);

    // initialize classifier
    toLog("trainClassifier - initialize classifier");
    weka.classifiers.functions.LinearRegression classifier = new weka.classifiers.functions.LinearRegression();

    // train classifier
    toLog("trainClassifier - train classifier");
    try {
      classifier.buildClassifier(curTrain);
    } catch (Exception ex) {
      // Logger.getLogger(MonolingualCoTrainingML.class.getName()).log(Level.SEVERE,
      // null, ex);
    }

    // evaluate classifier
    toLog("trainClassifier - evaluateClassifier");
    evaluateClassifier(classifier, language, fold, iteration, curTrain);

    return classifier;
  }

  private static Instances loadDataset(String path, String datatype) {
    Instances dataset = null;
    try {
      DataSource setsource = new DataSource(path);
      System.out.println("loading " + datatype + " file...");
      dataset = new Instances(setsource.getDataSet());
    } catch (java.lang.Exception e) {
      System.out.println("Unable to read file " + path);
    }
    return dataset;
  }

  private static void evaluateClassifier(Classifier classifier,
      String language, int fold, int iteration, Instances train) {
    Instances testset = loadDataset(testdatapath + "label-" + FS + TYPE
        + "-test-" + language + ".arff", "labeled-test");
    testset.setClassIndex(testset.numAttributes() - 1);

    int correct = 0;
    int total = testset.numInstances();

    double accuracy = 0;
    double precision = 0;
    double recall = 0;
    double fmeasure = 0;

    for (int i = 0; i < CLASSES.length; i++) {
      ClassEvaluation myClassEvaluation = new ClassEvaluation();
      double reference_class = CLASSES[i];

      for (Enumeration e = testset.enumerateInstances(); e.hasMoreElements();) {
        Instance inst = (Instance) e.nextElement();
        inst.setDataset(testset);
        double true_class = inst.classValue();
        double classification_prediction = 0;
        try {
          classification_prediction = classifier.classifyInstance(inst);
        } catch (Exception e1) {
          e1.printStackTrace();
        }
        // if (classificationPrediction <= objRegThreshold){
        if (classification_prediction < 0.5) {
          classification_prediction = 0;
        }
        // else if (classificationPrediction >= subjRegThreshold){
        else if (classification_prediction >= 0.5) {
          classification_prediction = 1;
        }

        // set actual_class
        boolean actual_class;
        if (reference_class == true_class) {
          actual_class = true;
        } else {
          actual_class = false;
        }

        // set predicted_class
        boolean predicted_class;
        if (reference_class == classification_prediction) {
          predicted_class = true;
        } else {
          predicted_class = false;
        }

        // add case to evaluation
        myClassEvaluation.addCase(predicted_class, actual_class);

        if (true_class == classification_prediction) {
          ++correct;
        }
      }

      // print output
      String results = "@ " + language + " " + iteration + "-" + fold
          + "-class " + reference_class + " bootstrapping P R F Acc\n"
          + myClassEvaluation.toString();
      System.out.println(results);
      try {
        output.write(results + "\n");
      } catch (IOException ex) {
        // Logger.getLogger(MonolingualRegressionTesting.class.getName()).log(Level.SEVERE,
        // null, ex);
      }

      // append measures to calculate overall macro P R F Acc
      precision += myClassEvaluation.getPrecision();
      recall += myClassEvaluation.getRecall();
      fmeasure += myClassEvaluation.getFmeasure();
      accuracy += myClassEvaluation.getAccuracy();
    }

    // average overall macro P R F Acc
    precision /= (double) CLASSES.length;
    recall /= (double) CLASSES.length;
    fmeasure /= (double) CLASSES.length;
    accuracy /= (double) CLASSES.length;

    String results = "@ " + language + " " + iteration + "-" + fold
        + "-overall bootstrapping P R F Acc\n" + precision + "\t" + recall
        + "\t" + fmeasure + "\t" + accuracy + "\t";

    System.out.println(results);
    try {
      output.write(results + "\n");
    } catch (IOException ex) {
      // Logger.getLogger(MonolingualRegressionTesting.class.getName()).log(Level.SEVERE,
      // null, ex);
    }
  }

  private static void toLog(String message) {
    if (debug == true) {
      // Date now = new Date();
      // System.out.println(DateFormat.getDateTimeInstance().format(now) + " ; "
      // + message);
      System.out.println(message);
    }
  }
}