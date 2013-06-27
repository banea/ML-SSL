import java.util.logging.Level;
import java.util.logging.Logger;
import weka.core.converters.ConverterUtils.DataSource;
import weka.core.Instances;
import weka.classifiers.Classifier;
import weka.core.converters.ArffLoader.ArffReader;
import weka.core.converters.ArffSaver;
import weka.core.Instance;
import weka.classifiers.Evaluation;
import java.io.*;
import java.util.*;
import java.text.DateFormat;

public class MultilingualCoTrainingRegression {
  // co-training using two regression classifiers at the same time

  private static final double DEVIATION = 0.4;
  private static final double subjRegThreshold = 1 - DEVIATION; // threshold has
                                                                // to be in the
                                                                // 0-1.0 range
  private static final double objRegThreshold = DEVIATION; // threshold has to
                                                           // be in the 0-1.0
                                                           // range
  private static final String datapath = "/local/carmen/MachineLearning/SenseLevel/ARFF/multilingual/regression/";
  private static String FS = "";
  // private static final String FS="fs-"; //feature selection (in case the arff
  // file name starts with some feature selection suffix
  // if nothing, then specify empty string
  // private static String sampleLabel="original"; //exactly the output of the
  // regression algorithm
  private static String sampleLabel = "mapped"; // the output of the regression
                                                // algorithm is mapped to 0,
  // if the label is closer to 0, or to 1, if it is closer to 1.
  private static final int NOOFEXAMPLES = 50;
  private static final int MAXFOLDS = 3;
  // !!!!modified the MAXITERATIONS number from 10 to 20 to continue from where
  // it left off
  private static final int MAXITERATIONS = 20;
  private static final String TYPE = "bal";
  private static final String[] LANGUAGES = { "ro" };
  private static final Double[] CLASSES = { 0.0, 1.0 }; // 0 - obj, 1 -subj
  private static BufferedWriter output;
  private static boolean debug = true;

  // private static boolean foundAgreement=false;

  /**
   * @param args
   */
  public static void main(String[] args) throws Exception {

    output = new BufferedWriter(new FileWriter(datapath + FS + "results-"
        + DEVIATION + "-" + NOOFEXAMPLES + ".txt"));

    // !!!!modified the fold number from 0 to 1 to continue from where it left
    // off
    for (int fold = 2; fold < MAXFOLDS; fold++) {

      // !!!!modified the iteration number from 0 to 11 to continue from where
      // it left off
      int iteration = 11;
      if (fold == 2) {
        iteration = 18;
      }

      for (; iteration <= MAXITERATIONS; iteration++) {

        int nextiteration = iteration + 1;

        toLog("load multilingual test data for iteration " + iteration);
        // load multilingual test data incrementally
        BufferedReader readerMult = new BufferedReader(new FileReader(datapath
            + FS + TYPE + "-test-" + LANGUAGES[0] + "-" + fold + "-"
            + iteration + ".arff"));
        ArffReader arffMult = new ArffReader(readerMult, 1000);
        Instances dataMult = arffMult.getStructure();
        dataMult.setClassIndex(dataMult.numAttributes() - 1);
        Instance instMult;

        toLog("opens multilingual Arff Savers for next iteration (i.e. #"
            + nextiteration + ") test data");
        // opens multilingual Arff Savers for next iteration test data
        ArffSaver saverMultTest = new ArffSaver();
        saverMultTest.setRetrieval(saverMultTest.INCREMENTAL);
        saverMultTest.setStructure(dataMult);
        saverMultTest.setFile(new File(datapath + FS + TYPE + "-test-"
            + LANGUAGES[0] + "-" + fold + "-" + nextiteration + ".arff"));

        toLog("entering trainClassifier function");
        // returns trained classifiers and evaluates classifier on labeled test
        // data
        Classifier multClassifier = trainClassifier(LANGUAGES[0], fold,
            iteration);

        toLog("stores the best performing n obj and n subj instance pairs");
        // stores the best performing n obj and n subj instance pairs
        // removes the maximum value from the PriorityQueue (which is stored in
        // the root)
        Comparator<InstanceValuePair> maxComparator = new MaximumValueComparator();
        PriorityQueue<InstanceValuePair> bestObj = new PriorityQueue<InstanceValuePair>(
            NOOFEXAMPLES, maxComparator);
        // removes the minimum value from the PriorityQueue (which is stored in
        // the root)
        Comparator<InstanceValuePair> minComparator = new MinimumValueComparator();
        PriorityQueue<InstanceValuePair> bestSubj = new PriorityQueue<InstanceValuePair>(
            NOOFEXAMPLES, minComparator);

        int testcount = 0;
        int instance_counter = 0;

        System.out.println("Evaluating test instances...");
        while ((instMult = arffMult.readInstance(dataMult)) != null) {
          instance_counter++;

          double label = -1.0;
          double multClsScore = multClassifier.classifyInstance(instMult);

          if (multClsScore < objRegThreshold) { // sample belongs to the
                                                // objective class
            if (sampleLabel.equals("mapped")) {
              label = 0.0; // objective score
            } else {
              label = multClsScore;
            }
          } else if (multClsScore > subjRegThreshold) { // sample belongs to the
                                                        // subjective class
            if (sampleLabel.equals("mapped")) {
              label = 1.0;
            } else {
              label = multClsScore; // subjective score
            }
          }

          InstanceValuePair to_be_considered_instance_value_pair = new InstanceValuePair(
              instMult, multClsScore, label);
          InstanceValuePair to_be_added_to_test_set_instance_value_pair = processInstances(
              to_be_considered_instance_value_pair, bestObj, bestSubj);

          if (to_be_added_to_test_set_instance_value_pair != null) {
            testcount++;
            // sets the dataset for the instances to be written and writes them
            // to the test files
            Instance testInstance = to_be_added_to_test_set_instance_value_pair
                .getInstance();
            testInstance.setDataset(dataMult);
            saverMultTest.writeIncremental(testInstance);
          }
        }

        toLog("Updating train set...");
        updateTrainSet(bestObj, bestSubj, fold, iteration);
        toLog("Number of test instances for iteration " + nextiteration + " :"
            + testcount);
        output.write("Number of test instances for iteration " + nextiteration
            + " :" + testcount + "\n");
        saverMultTest.writeIncremental(null);
        toLog("Found objective examples: " + bestObj.size()
            + " Found subjective examples: " + bestSubj.size() + "\n");
        if ((bestObj.size() < NOOFEXAMPLES) || (bestSubj.size() < NOOFEXAMPLES)) {
          toLog("Less than n examples found; pursue to next fold");
          break;
        }
      }
    }
    output.close();
  }

  private static void toLog(String message) {
    if (debug == true) {
      Date now = new Date();
      System.out.println(DateFormat.getDateTimeInstance().format(now) + " ; "
          + message);
    }
  }

  private static String toStringPriorityQueue(
      PriorityQueue<InstanceValuePair> best) {
    Iterator myPriorityQueueIterator = best.iterator();

    String myToString = "";

    while (myPriorityQueueIterator.hasNext()) {
      InstanceValuePair myInstanceValuePair = (InstanceValuePair) myPriorityQueueIterator
          .next();
      myToString += myInstanceValuePair.toString();
    }
    return myToString;
  }

  private static InstanceValuePair processInstances(
      InstanceValuePair to_be_considered_instance_value_pair,
      PriorityQueue<InstanceValuePair> bestObj,
      PriorityQueue<InstanceValuePair> bestSubj) {
    InstanceValuePair to_be_added_to_test_set_instance_value_pair = null;

    if (to_be_considered_instance_value_pair.getValue() == -1.0) { // if an
                                                                   // agreement
                                                                   // was not
                                                                   // reached,
                                                                   // return
                                                                   // instance
                                                                   // pair to be
                                                                   // added to
                                                                   // the next
                                                                   // test set
      to_be_added_to_test_set_instance_value_pair = to_be_considered_instance_value_pair;
    } else {
      // System.out.println(to_be_considered_instance_value_pair.getValue());
      if (to_be_considered_instance_value_pair.getLabel() <= objRegThreshold) { // i.e.
                                                                                // obj
        bestObj.add(to_be_considered_instance_value_pair);
        if (bestObj.size() > NOOFEXAMPLES) {
          to_be_added_to_test_set_instance_value_pair = bestObj.remove();
        }
      } else { // i.e. subj
        bestSubj.add(to_be_considered_instance_value_pair);
        if (bestSubj.size() > NOOFEXAMPLES) {
          to_be_added_to_test_set_instance_value_pair = bestSubj.remove();
        }
      }
    }
    return to_be_added_to_test_set_instance_value_pair;
  }

  private static void updateTrainSet(PriorityQueue<InstanceValuePair> bestObj,
      PriorityQueue<InstanceValuePair> bestSubj, int fold, int iteration)
      throws Exception {

    // load train data incrementally
    // English
    BufferedReader readerMult = new BufferedReader(new FileReader(datapath + FS
        + TYPE + "-train-" + LANGUAGES[0] + "-" + fold + "-" + iteration
        + ".arff"));
    ArffReader arffMult = new ArffReader(readerMult, 1000);
    Instances dataMult = arffMult.getStructure();
    dataMult.setClassIndex(dataMult.numAttributes() - 1);
    Instance instMult;

    int nextiteration = iteration + 1;
    // opens multilingual Arff Savers for next iteration train data
    ArffSaver saverMultTrain = new ArffSaver();
    saverMultTrain.setRetrieval(saverMultTrain.INCREMENTAL);
    saverMultTrain.setStructure(dataMult);
    saverMultTrain.setFile(new File(datapath + FS + TYPE + "-train-"
        + LANGUAGES[0] + "-" + fold + "-" + nextiteration + ".arff"));

    int count = 0;
    // write train data incrementally - first copy the current train data
    while ((instMult = arffMult.readInstance(dataMult)) != null) {
      count++;
      saverMultTrain.writeIncremental(instMult);
    }

    // write best test examples classified incrementally
    Iterator[] iteratorArray = new Iterator[2];
    iteratorArray[0] = bestObj.iterator();
    iteratorArray[1] = bestSubj.iterator();

    for (Iterator itr : iteratorArray) {
      while (itr.hasNext()) {
        count++;
        InstanceValuePair myInstanceValuePair = (InstanceValuePair) itr.next();

        // write English instance
        Instance myInstance = myInstanceValuePair.getInstance();
        myInstance.setDataset(dataMult);
        myInstance.setClassValue(myInstanceValuePair.getLabel());
        saverMultTrain.writeIncremental(myInstance);
      }
    }

    // close files
    saverMultTrain.writeIncremental(null);
    readerMult.close();
    System.out.println("Number of train instances for iteration "
        + nextiteration + " :" + count);
    output.write("Number of train instances for iteration " + nextiteration
        + " :" + count + "\n");
  }

  private static Classifier trainClassifier(String language, int fold,
      int iteration) {

    toLog("trainClassifier - starts loading data");
    // load train data
    Instances curTrain = loadDataset(datapath + FS + TYPE + "-train-"
        + language + "-" + fold + "-" + iteration + ".arff", "train");
    // set class attribute
    curTrain.setClassIndex(curTrain.numAttributes() - 1);

    toLog("trainClassifier - initialize classifier");
    // initialize classifier
    weka.classifiers.functions.LinearRegression classifier = new weka.classifiers.functions.LinearRegression();

    /*
     * weka.classifiers.functions.SMOreg classifier = new
     * weka.classifiers.functions.SMOreg(); try {
     * classifier.setOptions(weka.core.Utils.splitOptions(
     * "-C 1.0 -N 0 -I \"weka.classifiers.functions.supportVector.RegSMOImproved -L 0.0010 -W 1 -P 1.0E-12 -T 0.0010 -V\" -K \"weka.classifiers.functions.supportVector.PolyKernel -C 250007 -E 1.0\""
     * )); } catch (Exception ex) { ex.toString();
     * Logger.getLogger(multilingual.class.getName()).log(Level.SEVERE, null,
     * ex); }
     */

    toLog("trainClassifier - train classifier");
    // train classifier
    try {
      classifier.buildClassifier(curTrain);
    } catch (Exception ex) {
      Logger.getLogger(MultilingualCoTrainingRegression.class.getName()).log(
          Level.SEVERE, null, ex);
    }

    // evaluate classifier
    toLog("trainClassifier - evaluateClassifier");
    evaluateClassifier(classifier, language, fold, iteration, curTrain);

    return classifier;
  }

  private static void evaluateClassifier(Classifier classifier,
      String language, int fold, int iteration, Instances train) {
    Instances testset = loadDataset(datapath + FS + "label-" + TYPE + "-test-"
        + language + "-" + fold + ".arff", "labeled-test");
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
      }

      // print output
      String results = "@ " + language + " " + iteration + "-" + fold
          + "-class " + reference_class + " bootstrapping P R F Acc\n"
          + myClassEvaluation.toString();
      System.out.println(results);
      try {
        output.write(results + "\n");
      } catch (IOException ex) {
        Logger.getLogger(MultilingualCoTrainingRegression.class.getName()).log(
            Level.SEVERE, null, ex);
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
      Logger.getLogger(MonolingualCoTrainingRegression.class.getName()).log(
          Level.SEVERE, null, ex);
    }
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
}
