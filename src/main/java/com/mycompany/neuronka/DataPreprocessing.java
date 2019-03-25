/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package com.mycompany.neuronka;

import com.panayotis.gnuplot.JavaPlot;
import com.panayotis.gnuplot.plot.DataSetPlot;
import com.panayotis.gnuplot.style.PlotStyle;
import com.panayotis.gnuplot.style.Style;
import java.io.BufferedReader;
import java.io.File;
import java.io.FileOutputStream;
import java.io.FileReader;
import java.io.IOException;
import java.io.PrintWriter;
import java.math.BigDecimal;
import java.math.RoundingMode;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Set;
import org.apache.commons.math3.stat.descriptive.DescriptiveStatistics;
import org.apache.commons.math3.stat.descriptive.moment.Mean;
import org.apache.commons.math3.stat.descriptive.moment.StandardDeviation;
import org.apache.commons.math3.stat.descriptive.moment.Variance;
import org.neuroph.core.*;
import org.neuroph.core.data.DataSet;
import org.neuroph.core.data.DataSetRow;
import org.neuroph.core.events.LearningEvent;
import org.neuroph.core.events.LearningEventListener;
import org.neuroph.nnet.*;
import org.neuroph.nnet.learning.BackPropagation;
import org.neuroph.nnet.learning.MomentumBackpropagation;
import org.neuroph.util.data.norm.MaxNormalizer;
import org.neuroph.util.data.norm.Normalizer;

/**
 *
 * @author Patrik
 */
public class DataPreprocessing implements LearningEventListener{
    
    private static Map<Double, Double> map;
    private static String sensor = "AK09918C";
    private static String processedDataFile = "processedData.csv";
    private static List<double[]> oneActivity;
    private static double[] oneActivityAxisValues;
    private static int milisecondsInFrame = 3000;
    private static int numberOfSamplesInFrame = 600;
    private static String csvSplitBy = "\t";
    
    //variables for feature calc
    private static double mean;
    
    // for evaluating classification result
    int total, correct, incorrect;
    
   // if output is greater then this value it is considered as walking
    float classificationThreshold = 0.5f;

    public static void main(String[] args){    
        /*
        // create new perceptron network 
        //NeuralNetwork neuralNetwork = new Perceptron(2,1); 
        MultiLayerPerceptron neuralNetwork = new MultiLayerPerceptron(2, 5, 5, 1);
 
        // create training set 
        DataSet trainingSet = new DataSet(2,1);        
        //DataSet trainingSet = DataSet.createFromFile(processedDataFile, 2, 1, csvSplitBy);
              
        try (BufferedReader br = new BufferedReader(new FileReader(processedDataFile))) {
            while ((line = br.readLine()) != null) {
                String[] zaznam = line.split(csvSplitBy);
                // add training data to training set
                trainingSet.addRow(new DataSetRow (new double[]{Double.parseDouble(zaznam[0]), Double.parseDouble(zaznam[1])}, new double[]{Double.parseDouble(zaznam[2])})); 
            }
            
            // learn the training set 
            neuralNetwork.learn(trainingSet); 
 
            System.out.println("Done training."); 
            System.out.println("Testing network..."); 
         
            testNeuralNetwork(neuralNetwork); 
            
            // save the trained network into file 
            neuralNetwork.save("or_perceptron.nnet");
        } catch (Exception e) {
            e.printStackTrace();
        } 
        */
        Map<String, String> subory = new HashMap<String,String>();
        subory.put("C:/Users/Patrik/Desktop/Bakalarka - SensorRecorder dáta/pokus/indora-1540362934669.csv", sensor);
        subory.put("C:/Users/Patrik/Desktop/Bakalarka - SensorRecorder dáta/pokus/indora-1540363314900.csv", sensor);
        subory.put("C:/Users/Patrik/Desktop/Bakalarka - SensorRecorder dáta/pokus/indora-1540363406576.csv", sensor);        
        subory.put("C:/Users/Patrik/Desktop/Bakalarka - SensorRecorder dáta/pokus/indora-1540484172540.csv", sensor);        
        subory.put("C:/Users/Patrik/Desktop/Bakalarka - SensorRecorder dáta/pokus/indora-1540484680716.csv", sensor);              
        subory.put("C:/Users/Patrik/Desktop/Bakalarka - SensorRecorder dáta/pokus/indora-1549012172677.csv", "ACCELEROMETER");
        subory.put("C:/Users/Patrik/Desktop/Bakalarka - SensorRecorder dáta/pokus/indora-1549021777198.csv", "ACCELEROMETER");        
        subory.put("C:/Users/Patrik/Desktop/Bakalarka - SensorRecorder dáta/pokus/indora-1549022025135.csv", "ACCELEROMETER");        
        subory.put("C:/Users/Patrik/Desktop/Bakalarka - SensorRecorder dáta/pokus/indora-1549022068213.csv", "ACCELEROMETER");                
        subory.put("C:/Users/Patrik/Desktop/Bakalarka - SensorRecorder dáta/pokus/indora-1549482603748.csv", "MPU6500 Acceleration Sensor");        
        subory.put("C:/Users/Patrik/Desktop/Bakalarka - SensorRecorder dáta/pokus/indora-1549482689471.csv", "MPU6500 Acceleration Sensor");
        subory.put("C:/Users/Patrik/Desktop/Bakalarka - SensorRecorder dáta/pokus/indora-1549541475108.csv", sensor);        
        subory.put("C:/Users/Patrik/Desktop/Bakalarka - SensorRecorder dáta/pokus/indora-1549541552672.csv", sensor);        
        subory.put("C:/Users/Patrik/Desktop/Bakalarka - SensorRecorder dáta/pokus/indora-1549541559653.csv", sensor);        
        subory.put("C:/Users/Patrik/Desktop/Bakalarka - SensorRecorder dáta/pokus/indora-1549541572516.csv", sensor);        
        subory.put("C:/Users/Patrik/Desktop/Bakalarka - SensorRecorder dáta/pokus/indora-1549541582979.csv", sensor);        
        subory.put("C:/Users/Patrik/Desktop/Bakalarka - SensorRecorder dáta/pokus/indora-1549541633702.csv", sensor);        
        subory.put("C:/Users/Patrik/Desktop/Bakalarka - SensorRecorder dáta/pokus/indora-1549541673057.csv", sensor);        
        subory.put("C:/Users/Patrik/Desktop/Bakalarka - SensorRecorder dáta/pokus/indora-1549541708344.csv", sensor);
        
        for (Map.Entry<String, String> entry : subory.entrySet()) {
            String key = entry.getKey();
            String value = entry.getValue();
            System.out.println("Subor: " + key + " a akcelerometer: " + value);
            processData(value,key);
        }        
        
        //processData(sensor, "C:/Users/Patrik/Desktop/Bakalarka - SensorRecorder dáta/pokus/indora-1540363406576.csv");
        
        /*
        try {
            JavaPlot p = new JavaPlot();
            double[][] data = null;
            data = filterCSVFileBySensorAndActivity(sensor, activity, csvFile);
            if (data == null) {
                System.out.println("Data array is empty!");
                return;
            }
            PlotStyle myPlotStyle = new PlotStyle();
            myPlotStyle.setStyle(Style.HISTEPS);
            myPlotStyle.setLineWidth(1);
            DataSetPlot s = new DataSetPlot(data);
            s.setTitle("Graf " + activity);
            s.setPlotStyle(myPlotStyle);
            p.addPlot(s);
            p.plot();
        } catch (Exception e) {
            e.printStackTrace();
            System.out.println("Zadana aktivita sa v dátach nenachadza!");
        }
        */
        
        //spustenie neuronky
        (new DataPreprocessing()).run();
    }
    
    public static double[][] filterCSVFileBySensorAndActivity(String sensor, String activity, String csvFile) {
	String line = "";
        map = new HashMap<Double, Double>();
        try (BufferedReader br = new BufferedReader(new FileReader(csvFile))) {
            while ((line = br.readLine()) != null) {
                String[] zaznam = line.split(csvSplitBy);
                for (int i = 0; i < zaznam.length; i++) {
                    if (zaznam[2].equals(sensor) && zaznam[1].equals(activity)) {
                        map.put(Double.valueOf(zaznam[0]),Double.valueOf(Math.sqrt(Double.parseDouble(zaznam[3]) * Double.parseDouble(zaznam[3])
                                + Double.parseDouble(zaznam[4]) * Double.parseDouble(zaznam[4])
                                + Double.parseDouble(zaznam[5]) * Double.parseDouble(zaznam[5]))));
                    }
                }
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
        double[][] data = new double[map.size()][2];
        Set entries = map.entrySet();
        Iterator entriesIterator = entries.iterator();
        int i = 0;
        while (entriesIterator.hasNext()) {
            Map.Entry mapping = (Map.Entry) entriesIterator.next();
            data[i][0] = (double) mapping.getKey();
            data[i][1] = (double) mapping.getValue();
            i++;
        }
        return data;
    }
    
    public static void processData(String sensor, String csvFile){
        String line = "";
	String cvsSplitBy = "\t";
        String currentActivity;
        String activityCode;
        try (BufferedReader br = new BufferedReader(new FileReader(csvFile))) {
            while ((line = br.readLine()) != null) {
                String[] zaznam = line.split(cvsSplitBy);
                currentActivity = zaznam[1];
                //in this expirement we only assume activity "walking" and "standing"
                if (zaznam[2].equals(sensor) && (currentActivity.equals("standing")||currentActivity.equals("walking"))) {
                    if (currentActivity.equals("walking")) {
                        activityCode = "1";
                    } else {
                        activityCode = "0";
                    }
                    oneActivity = new ArrayList<>();
                    oneActivity.add(new double[]{Double.valueOf(zaznam[0]), Math.sqrt(Double.parseDouble(zaznam[3])*Double.parseDouble(zaznam[3])
                                        + Double.parseDouble(zaznam[4])*Double.parseDouble(zaznam[4])
                                        + Double.parseDouble(zaznam[5])*Double.parseDouble(zaznam[5])),Double.parseDouble(activityCode)});
                    while ((line = br.readLine()) != null) {
                        zaznam = line.split(cvsSplitBy);
                        if (zaznam[1].equals(currentActivity)) {
                            if (zaznam[2].equals(sensor)) {                               
                                //we compute 3 axes together like (Math.sqrt(x*x+y*y+z*z)) 
                                oneActivity.add(new double[]{Double.valueOf(zaznam[0]), Math.sqrt(Double.parseDouble(zaznam[3])*Double.parseDouble(zaznam[3])
                                        + Double.parseDouble(zaznam[4])*Double.parseDouble(zaznam[4])
                                        + Double.parseDouble(zaznam[5])*Double.parseDouble(zaznam[5])),Double.parseDouble(activityCode)});
                            }                            
                        } else {
                            //tu spracovat data aktivity a dat do suboru
                            System.out.println("pocet zaznamov aktivity " + currentActivity + "  :" + oneActivity.size());
                            
                            //len na histogramy
                            //if (currentActivity.equals("standing")) {
                                compute(oneActivity); 
                            //}
                            
                            oneActivity = new ArrayList<>();
                            break;
                        }
                    } 
                    //ak sme došli na koniec suboru a posledna aktivita vo file bola este walking/standing tak ju spracujeme
                    if (oneActivity.size() != 0) {
                        System.out.println("pocet zaznamov aktivity " + currentActivity + "  :" + oneActivity.size());
                        
                        //len na histogramy
                        //if (currentActivity.equals("standing")) {
                           compute(oneActivity); 
                        //}
                        
                        oneActivity = new ArrayList<>();
                    }
                }
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
    }    

    //here we should divide activity into windows
    //and compute our features for every window and insert one line with feature values into dataset file
    private static void compute(List<double[]> activity) {
        PrintWriter pw = null;
        //int numberOfSamples = activity.size();
        int startTime = (int)activity.get(0)[0]; 
        int endTime = (int)activity.get(activity.size()-1)[0];
        int currentStartListPosition = 0;

        if (endTime-startTime<milisecondsInFrame) {
            return;
        }
        
        int frameEndTime = startTime + milisecondsInFrame;
        
        try {
            pw = new PrintWriter(new FileOutputStream(new File(processedDataFile),true /*append*/));
            while (frameEndTime <= endTime) {
                //here we compute window containing samples from between startTime and frameEndTime
                //compute features
                
                //get oneActivityAxisValues for current frame
                int i = currentStartListPosition;
                while (oneActivity.get(i)[0]<frameEndTime) {
                    i++;                    
                }                
                oneActivityAxisValues = new double[i-currentStartListPosition];
                for (int j = 0; j < oneActivityAxisValues.length; j++) {
                    oneActivityAxisValues[j]=oneActivity.get(currentStartListPosition+j)[1];
                }
                
                //insert into dataset
                pw.print(mean() + csvSplitBy + standardDeviation() + csvSplitBy 
                        + variance() + csvSplitBy + meanAbsoluteDeviation() + csvSplitBy 
                        + rootMeanSquare() + csvSplitBy + interquartileRange() + csvSplitBy
                        + (int)activity.get(0)[2]);
                pw.println();
                
                //update variables currentStartListPosition,startTime,frameEndTime
                int newFrameStartTime = startTime + milisecondsInFrame/2; //diveded by 2 because we have 50% window overlapping
                while (oneActivity.get(currentStartListPosition)[0]<newFrameStartTime) {                    
                    currentStartListPosition++;
                }
                startTime = (int) oneActivity.get(currentStartListPosition)[0];
                frameEndTime = startTime + milisecondsInFrame; 
            }         
        } catch (Exception e) {
            e.printStackTrace();
        } finally {
            if(pw != null){
                pw.close();
            }
        }
    }
    
    //metody na vypocty featurov, ktore sa budu volat v compute metode a na vstupe budu bat oneActivity
    
    //simple mean
    private static double mean(){
        mean = new Mean().evaluate(oneActivityAxisValues);
        return mean;
    }
    
    //standard deviation
    private static double standardDeviation(){
        return new StandardDeviation().evaluate(oneActivityAxisValues, mean);
    }
    
    //variance
    private static double variance(){
        return new Variance().evaluate(oneActivityAxisValues, mean);
    }
    
    //mean absolute deviation
    private static double meanAbsoluteDeviation(){
        double sum = 0.0;
        for (int i = 0; i < oneActivityAxisValues.length; i++) {
            sum = sum + Math.abs(oneActivityAxisValues[i] - mean);
        }
        return sum/oneActivityAxisValues.length;
    }
    
    //root mean square
    private static double rootMeanSquare(){
        double sum = 0.0;
        for (int i = 0; i < oneActivityAxisValues.length; i++) {
            sum = sum + Math.pow(oneActivityAxisValues[i],2);
        }
        return Math.sqrt(sum/oneActivityAxisValues.length);
    }
    
    //interquartile range
    private static double interquartileRange(){
        DescriptiveStatistics ds = new DescriptiveStatistics(oneActivityAxisValues);
        return ds.getPercentile(75) - ds.getPercentile(25);
    }
    
    /*
        NEURAL NETWORK
    */
    
    public void run() {
        System.out.println("Creating training and test set from file...");
        int inputsCount = 6;
        int outputsCount = 1;
        
        //Create data set from file
        DataSet dataSet = DataSet.createFromFile(processedDataFile, inputsCount, outputsCount, csvSplitBy);
        dataSet.shuffle();

        //Normalizing data set
        Normalizer normalizer = new MaxNormalizer();
        normalizer.normalize(dataSet);
        
        //save normalized dataset to file 
        //dataSet.saveAsTxt("dataset.csv", csvSplitBy);

        //Creatinig training set (70%) and test set (30%)
        List<DataSet> trainingAndTestSet = dataSet.split(70, 30);
        DataSet trainingSet = trainingAndTestSet.get(0);
        DataSet testSet = trainingAndTestSet.get(1);

        //Create MultiLayerPerceptron neural network
        MultiLayerPerceptron neuralNet = new MultiLayerPerceptron(inputsCount, 16, outputsCount);

        //attach listener to learning rule
        MomentumBackpropagation learningRule = (MomentumBackpropagation) neuralNet.getLearningRule();
        learningRule.addListener(this);
        
        learningRule.setLearningRate(0.3);
        learningRule.setMaxError(0.01);
        learningRule.setMaxIterations(9000);

        System.out.println("Training network...");
        //train the network with training set
        neuralNet.learn(trainingSet);

        System.out.println("Testing network...");
        testNeuralNetwork(neuralNet, testSet);
    }
    
    public void testNeuralNetwork(NeuralNetwork neuralNet, DataSet testSet) {
        System.out.println("********************** TEST RESULT **********************");
        for (DataSetRow testSetRow : testSet.getRows()) {
            neuralNet.setInput(testSetRow.getInput());
            neuralNet.calculate();
            
            // get network output
            double[] networkOutput = neuralNet.getOutput();
            int predicted = interpretOutput(networkOutput);

            // get target/desired output
            double[] desiredOutput = testSetRow.getDesiredOutput();
            int target = (int)desiredOutput[0];
            
            // count predictions
            countPredictions(predicted, target);
        }
        
        System.out.println("Total cases: " + total + ". ");
        System.out.println("Correctly predicted cases: " + correct);
        System.out.println("Incorrectly predicted cases: " + incorrect);
        double percentTotal = (correct / (double)total) * 100;
        System.out.println("Predicted correctly: " + formatDecimalNumber(percentTotal) + "%. ");
    }
    
     @Override
    public void handleLearningEvent(LearningEvent event) {
        BackPropagation bp = (BackPropagation) event.getSource();
        if (event.getEventType().equals(LearningEvent.Type.LEARNING_STOPPED)) {
            double error = bp.getTotalNetworkError();
            System.out.println("Training completed in " + bp.getCurrentIteration() + " iterations, ");
            System.out.println("With total error: " + formatDecimalNumber(error));
        } else {
            System.out.println("Iteration: " + bp.getCurrentIteration() + " | Network error: " + bp.getTotalNetworkError());
        }
    }

    public int interpretOutput(double[] array) {
        if (array[0] >= classificationThreshold) {
            return 1;
        }else {
            return 0;
        }
    }

    public void countPredictions(int prediction, int target) {
        if (prediction == target) {
            correct++;
        } else {
            incorrect++;
        }
        total++;
    }

    //Formating decimal number to have 3 decimal places
    public String formatDecimalNumber(double number) {
        return new BigDecimal(number).setScale(4, RoundingMode.HALF_UP).toString();
    }
}
