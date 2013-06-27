// co-training using the confidence of both classifiers at the same time

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

public class MonolingualCoTrainingML {

    private static final double CONFIDENCE = 0.53; //confidence has to be in the 0-1.0 range
    private static final String datapath = "/local/carmen/MachineLearning/SenseLevel/ARFF/monolingual/classification/";
    private static String FS="";
    //private static final String FS="fs-"; //feature selection (in case the arff file name starts with some feature selection suffix
    //if nothing, then specify empty string 
    private static final int NOOFEXAMPLES = 3;
    private static final int MAXFOLDS = 3;
    private static final int MAXITERATIONS = 10;
    private static final String TYPE = "bal";
    private static final String[] LANGUAGES = {"en", "ro"};
    private static final Double[] CLASSES = {0.0, 1.0}; //0 - obj, 1 -subj
    private static BufferedWriter output;
    //private static boolean foundAgreement=false;
    

    /**
     * @param args
     */
    public static void main(String[] args) throws Exception {
        
        output = new BufferedWriter(new FileWriter (datapath+FS+"results-"+CONFIDENCE+"-"+NOOFEXAMPLES+".txt"));

        for (int fold = 0; fold < MAXFOLDS; fold++) {
            for (int iteration=0; iteration <= MAXITERATIONS; iteration++){
            	
            	int nextiteration = iteration + 1;

                //load test data incrementally
                //English
                BufferedReader readerEn = new BufferedReader(new FileReader(datapath +FS+TYPE+"-test-" + LANGUAGES[0] + "-" + fold + "-" + iteration + ".arff"));
                ArffReader arffEn = new ArffReader(readerEn, 1000);
                Instances dataEn = arffEn.getStructure();
                dataEn.setClassIndex(dataEn.numAttributes() - 1);
                Instance instEn;

                //Romanian
                BufferedReader readerRo = new BufferedReader(new FileReader(datapath +FS+TYPE+"-test-" + LANGUAGES[1] + "-" + fold + "-" + iteration + ".arff"));
                ArffReader arffRo = new ArffReader(readerRo, 1000);
                Instances dataRo = arffRo.getStructure();
                dataRo.setClassIndex(dataRo.numAttributes() - 1);
                Instance instRo;

                //opens Arff Savers for next iteration test data
                //English
                ArffSaver saverEnTest = new ArffSaver();
                saverEnTest.setRetrieval(saverEnTest.INCREMENTAL);
                saverEnTest.setStructure(dataEn);
                saverEnTest.setFile(new File(datapath + FS + TYPE +"-test-" + LANGUAGES[0] + "-" + fold + "-" + nextiteration + ".arff"));
                //Romanian
                ArffSaver saverRoTest = new ArffSaver();
                saverRoTest.setRetrieval(saverRoTest.INCREMENTAL);
                saverRoTest.setStructure(dataRo);
                saverRoTest.setFile(new File(datapath +FS+TYPE+"-test-" + LANGUAGES[1] + "-" + fold + "-" + nextiteration + ".arff"));

                //returns trained classifiers and evaluates classifier on labeled test data
                Classifier enClassifier = trainClassifier(LANGUAGES[0], fold, iteration);
                Classifier roClassifier = trainClassifier(LANGUAGES[1], fold, iteration);
                
                //stores the best performing n obj and n subj instance pairs
                //removes the minimum value from the PriorityQueue (which is stored in the root)
                Comparator<InstanceValuePair> minComparator = new MinimumValueComparator();
                PriorityQueue<InstanceValuePair> bestObj = new PriorityQueue<InstanceValuePair>(NOOFEXAMPLES, minComparator);
                PriorityQueue<InstanceValuePair> bestSubj = new PriorityQueue<InstanceValuePair>(NOOFEXAMPLES, minComparator);
                
                
                int testcount = 0;
                int instance_counter = 0;

                System.out.println("Evaluating test instances...");
                while ((instEn = arffEn.readInstance(dataEn)) != null && (instRo = arffRo.readInstance(dataRo)) != null) {
                	instance_counter++;
                	                	
                    double[] enClsLabel = enClassifier.distributionForInstance(instEn);
                    double[] roClsLabel = roClassifier.distributionForInstance(instRo);
                    
                    double average_score = -1;
                    double label=-1;
                    for (int j = 0; j < CLASSES.length; j++) {
                        if (enClsLabel[j] > CONFIDENCE && roClsLabel[j] > CONFIDENCE) {
                            average_score = (enClsLabel[j] + roClsLabel[j])/2;
                            label=(double)j;
                        }
                    }
                    
                    
                    InstanceValuePair to_be_considered_instance_value_pair = new InstanceValuePair(instEn, instRo, average_score, label);
                    InstanceValuePair to_be_added_to_test_set_instance_value_pair = processInstances (to_be_considered_instance_value_pair, bestObj, bestSubj);
                    
              
                    if (to_be_added_to_test_set_instance_value_pair != null){
                    	if (label == 0){
                        	System.out.println("Instance "+instance_counter+" enClsLabel[0]="+enClsLabel[0]+
                    		" enClsLabel[1]="+enClsLabel[1]+" roClsLabel[0]="+roClsLabel[0]+" roClsLabel[1]="+roClsLabel[1]);
                        	System.out.println(to_be_considered_instance_value_pair.getValue());
                            System.out.println(to_be_added_to_test_set_instance_value_pair.getValue());
                            System.out.println("Best subj priority queue: "+toStringPriorityQueue(bestSubj));
                            System.out.println("Best obj priority queue: "+toStringPriorityQueue(bestObj));
                        }
                    	testcount++;
                    	//sets the dataset for the instances to be written and writes them to the test files
                    	Instance testEnInstance = to_be_added_to_test_set_instance_value_pair.getLang1Instance();
                    	testEnInstance.setDataset(dataEn);
                    	saverEnTest.writeIncremental(testEnInstance);
                    	
                    	Instance testRoInstance = to_be_added_to_test_set_instance_value_pair.getLang2Instance();
                    	testRoInstance.setDataset(dataRo);
                    	saverRoTest.writeIncremental(testRoInstance);    
                    }
                }

                System.out.println("Updating train set...");
                updateTrainSet(bestObj, bestSubj, fold, iteration);
                System.out.println("Number of test instances for iteration "+nextiteration+" :"+testcount);
                output.write("Number of test instances for iteration "+nextiteration+" :"+testcount+"\n");
                saverEnTest.writeIncremental(null);
                saverRoTest.writeIncremental(null);
                System.out.println("Found objective examples: "+bestObj.size()+" Found subjective examples: "+bestSubj.size()+"\n");
                if ((bestObj.size() < NOOFEXAMPLES) || (bestSubj.size() < NOOFEXAMPLES)){break;}
            } 
        }
        output.close();
    }
    
    private static String toStringPriorityQueue(PriorityQueue<InstanceValuePair> best){
    	Iterator myPriorityQueueIterator = best.iterator();
    	
    	String myToString = "";
    	
        while(myPriorityQueueIterator.hasNext()){
               	InstanceValuePair myInstanceValuePair = (InstanceValuePair) myPriorityQueueIterator.next();
               	myToString+=myInstanceValuePair.toString();
               	}
        return myToString;
    }
    
    private static InstanceValuePair processInstances(InstanceValuePair to_be_considered_instance_value_pair, PriorityQueue<InstanceValuePair> bestObj, PriorityQueue<InstanceValuePair> bestSubj){
    	InstanceValuePair to_be_added_to_test_set_instance_value_pair = null;
    	    	      
        if (to_be_considered_instance_value_pair.getValue() == -1.0) { //if an agreement was not reached, return instance pair to be added to the next test set
        	to_be_added_to_test_set_instance_value_pair = to_be_considered_instance_value_pair;
        } else {
        	//System.out.println(to_be_considered_instance_value_pair.getValue());
        	if (to_be_considered_instance_value_pair.getLabel() == 0 ) { //i.e. obj
        		bestObj.add(to_be_considered_instance_value_pair);
        		if (bestObj.size() > NOOFEXAMPLES){
        			to_be_added_to_test_set_instance_value_pair = bestObj.remove();
    			}
    		} else { //i.e. subj
    			bestSubj.add(to_be_considered_instance_value_pair);
    			if (bestSubj.size() > NOOFEXAMPLES){
    				to_be_added_to_test_set_instance_value_pair = bestSubj.remove();
    			}
    		}
        }
    	return to_be_added_to_test_set_instance_value_pair;
    }
    
    private static void updateTrainSet(PriorityQueue<InstanceValuePair> bestObj, PriorityQueue<InstanceValuePair> bestSubj, int fold, int iteration) throws Exception {

        //load train data incrementally
        //English
        BufferedReader readerEn = new BufferedReader(new FileReader(datapath +FS+ TYPE +"-train-" + LANGUAGES[0] + "-" + fold + "-" + iteration + ".arff"));
        ArffReader arffEn = new ArffReader(readerEn, 1000);
        Instances dataEn = arffEn.getStructure();
        dataEn.setClassIndex(dataEn.numAttributes() - 1);
        Instance instEn;
        //Romanian
        BufferedReader readerRo = new BufferedReader(new FileReader(datapath +FS+ TYPE +"-train-" + LANGUAGES[1] + "-" + fold + "-" + iteration + ".arff"));
        ArffReader arffRo = new ArffReader(readerRo, 1000);
        Instances dataRo = arffRo.getStructure();
        dataRo.setClassIndex(dataRo.numAttributes() - 1);
        Instance instRo;

        int nextiteration = iteration + 1;
        //opens Arff Savers for next iteration train data
        //English
        ArffSaver saverEnTrain = new ArffSaver();
        saverEnTrain.setRetrieval(saverEnTrain.INCREMENTAL);
        saverEnTrain.setStructure(dataEn);
        saverEnTrain.setFile(new File(datapath +FS +TYPE+"-train-" + LANGUAGES[0] + "-" + fold + "-" + nextiteration + ".arff"));
        //Romanian
        ArffSaver saverRoTrain = new ArffSaver();
        saverRoTrain.setRetrieval(saverRoTrain.INCREMENTAL);
        saverRoTrain.setStructure(dataRo);
        saverRoTrain.setFile(new File(datapath +FS+TYPE+"-train-" + LANGUAGES[1] + "-" + fold + "-" + nextiteration + ".arff"));

	int count=0;
        //write train data incrementally - first copy the current train data
        while ((instEn = arffEn.readInstance(dataEn)) != null && (instRo = arffRo.readInstance(dataRo)) != null) {
	    count++;
            saverEnTrain.writeIncremental(instEn);
            saverRoTrain.writeIncremental(instRo);
        }

        //write best test examples classified incrementally
        Iterator[] iteratorArray = new Iterator[2];
        iteratorArray[0] = bestObj.iterator();
        iteratorArray[1] = bestSubj.iterator();
        
        for (Iterator itr : iteratorArray) {
       	 while(itr.hasNext()){
            	count++;
            	InstanceValuePair myInstanceValuePair = (InstanceValuePair) itr.next();
            	
            	//write English instance
            	Instance enInstance = myInstanceValuePair.getLang1Instance();
            	enInstance.setDataset(dataEn);
            	enInstance.setClassValue(myInstanceValuePair.getLabel());
            	saverEnTrain.writeIncremental(enInstance);
            	//write Romanian instance
            	Instance roInstance = myInstanceValuePair.getLang2Instance();
            	roInstance.setDataset(dataRo);
            	roInstance.setClassValue(myInstanceValuePair.getLabel());
            	saverRoTrain.writeIncremental(roInstance);
            	}
        }

        //close files
        saverEnTrain.writeIncremental(null);
        saverRoTrain.writeIncremental(null);
        readerEn.close();
        readerRo.close();
        System.out.println("Number of train instances for iteration "+nextiteration+" :"+count);
        output.write("Number of train instances for iteration "+nextiteration+" :"+count+"\n");
    }

    private static Classifier trainClassifier(String language, int fold, int iteration) {

        //load train data
        Instances curTrain = loadDataset(datapath +FS+TYPE+"-train-" + language + "-" + fold + "-" + iteration + ".arff");
        //set class attribute
        curTrain.setClassIndex(curTrain.numAttributes() - 1);

        //initialize classifier
        //weka.classifiers.bayes.NaiveBayes classifier = new weka.classifiers.bayes.NaiveBayes();

        
        weka.classifiers.functions.LibSVM classifier = new weka.classifiers.functions.LibSVM();
        try {
        	classifier.setOptions(weka.core.Utils.splitOptions("weka.classifiers.functions.LibSVM -S 0 -K 2 -D 3 -G 0.0 -R 0.0 -N 0.5 -M 40.0 -C 1.0 -E 0.0010 -P 0.1"));
        } catch (Exception ex) {
            Logger.getLogger(MonolingualCoTrainingML.class.getName()).log(Level.SEVERE, null, ex);
        }
        classifier.setProbabilityEstimates(true);
        

        //train classifier
        try {
            classifier.buildClassifier(curTrain);
        } catch (Exception ex) {
            Logger.getLogger(MonolingualCoTrainingML.class.getName()).log(Level.SEVERE, null, ex);
        }

        //evaluate classifier
        evaluateClassifier(classifier, language, fold, iteration, curTrain);

        return classifier;
    }

    private static void evaluateClassifier(Classifier classifier, String language, int fold, int iteration, Instances train) {
        Instances testset = loadDataset(datapath +FS+"label-"+TYPE+"-test-" + language + "-"+fold+".arff");
        testset.setClassIndex(testset.numAttributes() - 1);

        Evaluation eval;
        try {
            eval = new Evaluation(train);
            eval.evaluateModel(classifier, testset);
            System.out.println(eval.toClassDetailsString("@ " + language + " " + iteration + "-" + fold + "-class bootstrapping"));
            System.out.println(eval.toSummaryString("@ " + language + " " + iteration + "-" + fold + "-overall bootstrapping", false));
            output.write(eval.toClassDetailsString("@ " + language + " " + iteration + "-" + fold + "-class bootstrapping")+"\n");
            output.write(eval.toSummaryString("@ " + language + " " + iteration + "-" + fold + "-overall bootstrapping", false)+"\n");
        } catch (Exception ex) {
            Logger.getLogger(MonolingualCoTrainingML.class.getName()).log(Level.SEVERE, null, ex);
        }
    }

    private static Instances loadDataset(String path) {
        Instances dataset = null;
        try {
            DataSource setsource = new DataSource(path);
            System.out.println("loading test file...");
            dataset = new Instances(setsource.getDataSet());
        } catch (java.lang.Exception e) {
            System.out.println("Unable to read file " + path);
        }
        return dataset;
    }
}
