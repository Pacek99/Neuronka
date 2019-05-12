/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package com.mycompany.neuronka;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileOutputStream;
import java.io.FileReader;
import java.io.IOException;
import java.io.PrintWriter;
import java.math.BigDecimal;
import java.math.BigInteger;
import java.math.RoundingMode;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Set;
import org.apache.commons.math3.complex.Complex;
import org.apache.commons.math3.stat.correlation.Covariance;
import org.apache.commons.math3.stat.descriptive.DescriptiveStatistics;
import org.apache.commons.math3.stat.descriptive.moment.Mean;
import org.apache.commons.math3.stat.descriptive.moment.StandardDeviation;
import org.apache.commons.math3.stat.descriptive.moment.Variance;
import org.apache.commons.math3.transform.DftNormalization;
import org.apache.commons.math3.transform.FastFourierTransformer;
import org.apache.commons.math3.transform.TransformType;
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
public class ExternalDatasetVersion implements LearningEventListener{
    
    private static Map<Double, Double> map;
    private static String sensor = "ST_LIS3DHTR";
    private static String processedDataFile = "processedDataExternalDataset.csv";
    private static List<double[]> oneActivity;
    private static double[] oneActivityAxisValues;
    
    private static double[] oneActivityXAxisValues;
    private static double[] oneActivityYAxisValues;
    private static double[] oneActivityZAxisValues;
    
    private static int milisecondsInFrame = 3000;
    private static int numberOfSamplesInFrame = 600;
    private static String csvSplitBy = ",";
    
    //variables for feature calc
    private static double mean;
       
   // if output is greater then this value it is considered as walking
    float classificationThreshold = 0.5f;
    
    //Important for evaluating network result
    public int[] count = new int[8];
    public int[] correct = new int[8];
    int unpredicted = 0;
    int[][] testingResults = new int[7][7];
    
    //variables for external dataset
    private static BigInteger nanosecondsInFrame = new BigInteger("3000000000");
    private static List<BigInteger> oneActivityTimes;
    
    //vypocet celkoveho casu datasetu
    private static double celkovyCas;

    public static void main(String[] args){ 
        Map<String, String> subory = new HashMap<String,String>();
        
        //nacitanie suborov na vytah
        subory.put("src/main/resources/Vytah/indora-1556082478032.csv", sensor);
        subory.put("src/main/resources/Vytah/indora-1556082506828.csv", sensor);
        subory.put("src/main/resources/Vytah/indora-1556082533389.csv", sensor);
        subory.put("src/main/resources/Vytah/indora-1556082560909.csv", sensor);
        subory.put("src/main/resources/Vytah/indora-1556110062122.csv", sensor);
        
        subory.put("src/main/resources/Vytah/indora-1556110357233.csv", sensor);
        subory.put("src/main/resources/Vytah/indora-1556110929890.csv", sensor);
        subory.put("src/main/resources/Vytah/indora-1556110955057.csv", sensor);
        subory.put("src/main/resources/Vytah/indora-1556110980942.csv", sensor);
        subory.put("src/main/resources/Vytah/indora-1556111007555.csv", sensor);
        
        subory.put("src/main/resources/Vytah/indora-1556111051918.csv", sensor);
        subory.put("src/main/resources/Vytah/indora-1556111077350.csv", sensor);
        subory.put("src/main/resources/Vytah/indora-1556111098861.csv", sensor);
        subory.put("src/main/resources/Vytah/indora-1556111125321.csv", sensor);
        subory.put("src/main/resources/Vytah/indora-1556111149569.csv", sensor);
        
        subory.put("src/main/resources/Vytah/indora-1556111174552.csv", sensor);
        subory.put("src/main/resources/Vytah/indora-1556111200762.csv", sensor);
        subory.put("src/main/resources/Vytah/indora-1556111227247.csv", sensor);
        subory.put("src/main/resources/Vytah/indora-1556111281572.csv", sensor);
        subory.put("src/main/resources/Vytah/indora-1556111304040.csv", sensor);
        
        subory.put("src/main/resources/Vytah/indora-1556111329156.csv", sensor);
        subory.put("src/main/resources/Vytah/indora-1556111352315.csv", sensor);
        subory.put("src/main/resources/Vytah/indora-1556111383811.csv", sensor);
        subory.put("src/main/resources/Vytah/indora-1556111407332.csv", sensor);
        subory.put("src/main/resources/Vytah/indora-1556111433500.csv", sensor);
        
        subory.put("src/main/resources/Vytah/indora-1556111459842.csv", sensor);
        subory.put("src/main/resources/Vytah/indora-1556111481240.csv", sensor);
        subory.put("src/main/resources/Vytah/indora-1556111504747.csv", sensor);
        subory.put("src/main/resources/Vytah/indora-1556111530481.csv", sensor);
        subory.put("src/main/resources/Vytah/indora-1556111618750.csv", sensor);
        
        subory.put("src/main/resources/Vytah/indora-1556111665614.csv", sensor);
        subory.put("src/main/resources/Vytah/indora-1549541522778.csv", sensor);        
        subory.put("src/main/resources/Vytah/indora-1549541647313.csv", sensor);                
        subory.put("src/main/resources/Vytah/indora-1540546792759.csv", sensor);
        subory.put("src/main/resources/Vytah/indora-1540554936805.csv", sensor);
        
        subory.put("src/main/resources/Vytah/indora-1554699869876.csv", sensor);
        subory.put("src/main/resources/Vytah/indora-1554699898743.csv", sensor);
        subory.put("src/main/resources/Vytah/indora-1554699925682.csv", sensor);
        subory.put("src/main/resources/Vytah/indora-1554699994623.csv", sensor);
        subory.put("src/main/resources/Vytah/indora-1554710761298.csv", sensor);
        
        subory.put("src/main/resources/Vytah/indora-1554728410434.csv", sensor);
        subory.put("src/main/resources/Vytah/indora-1554728381607.csv", sensor);     
        
        celkovyCas = 0;
        
        // vygenerovanie datasetu pre vytah         
        for (Map.Entry<String, String> entry : subory.entrySet()) {
            String key = entry.getKey();
            String value = entry.getValue();
            System.out.println("Subor: " + key + " a akcelerometer: " + value);
            processDataVytah(value,key);
        }
        
        
        System.out.println("Celkovy cas: " + celkovyCas);
        celkovyCas = 0;
        
        //skuska s datasetom z internetu
        processData("src/main/resources/Phones_accelerometer.csv");
        
        //vypis celkoveho casu
        System.out.println("Celkovy cas: " + celkovyCas);
       
        //spustenie neuronky
        //(new ExternalDatasetVersion()).run();
    }
    public static void processDataVytah(String sensor, String csvFile){
        String line = "";
	String cvsSplitBy = "\t";
        String currentActivity;
        String activityCode = "";
        try (BufferedReader br = new BufferedReader(new FileReader(csvFile))) {
            while ((line = br.readLine()) != null) {
                String[] zaznam = line.split(cvsSplitBy);
                currentActivity = zaznam[1];
                
                if (zaznam[2].equals(sensor)) {                                 
                    if (currentActivity.equals("elevatorUp")) {
                        activityCode = "5";
                    }
                    if (currentActivity.equals("elevatorDown")) {
                        activityCode = "6";
                    }                    
                   
                    oneActivity = new ArrayList<>();
                    //add values for time, xAxis, yAxis, zAxis, activityCode
                    oneActivity.add(new double[]{Double.valueOf(zaznam[0]), Double.parseDouble(zaznam[3]), Double.parseDouble(zaznam[4]),
                                    Double.parseDouble(zaznam[5]), Double.parseDouble(activityCode)});
                    while ((line = br.readLine()) != null) {
                        zaznam = line.split(cvsSplitBy);
                        if (zaznam[1].equals(currentActivity)) {
                            if (zaznam[2].equals(sensor)) {  
                                //add values for time, xAxis, yAxis, zAxis, activityCode
                                oneActivity.add(new double[]{Double.valueOf(zaznam[0]), Double.parseDouble(zaznam[3]), Double.parseDouble(zaznam[4]),
                                    Double.parseDouble(zaznam[5]), Double.parseDouble(activityCode)});
                            }                            
                        } else {
                            //tu spracovat data aktivity a dat do suboru
                            System.out.println("pocet zaznamov aktivity " + currentActivity + "  :" + oneActivity.size());
                            
                            //len na histogramy
                            //if (currentActivity.equals("elevatorDown")) {
                                computeVytah(oneActivity); 
                            //}
                            
                            oneActivity = new ArrayList<>();
                            break;
                        }
                    } 
                    //ak sme došli na koniec suboru a posledna aktivita vo file bola este walking/standing tak ju spracujeme
                    if (oneActivity.size() != 0) {
                        System.out.println("pocet zaznamov aktivity " + currentActivity + "  :" + oneActivity.size());
                        
                        //len na histogramy
                        //if (currentActivity.equals("elevatorDown")) {
                           computeVytah(oneActivity); 
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
    private static void computeVytah(List<double[]> activity) {
        celkovyCas += (activity.get(activity.size()-1)[0] - activity.get(0)[0])/1000;
        int numberOfFramesComputed = 0;
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
                
                //oneActivityAxisValues = new double[i-currentStartListPosition];
                oneActivityXAxisValues = new double[i-currentStartListPosition];
                oneActivityYAxisValues = new double[i-currentStartListPosition];
                oneActivityZAxisValues = new double[i-currentStartListPosition];
                                                
                for (int j = 0; j < oneActivityXAxisValues.length; j++) {
                    //oneActivityAxisValues[j]=oneActivity.get(currentStartListPosition+j)[1];
                    oneActivityXAxisValues[j]=oneActivity.get(currentStartListPosition+j)[1];
                    oneActivityYAxisValues[j]=oneActivity.get(currentStartListPosition+j)[2];
                    oneActivityZAxisValues[j]=oneActivity.get(currentStartListPosition+j)[3];
                }
                
                String outputValues = "";
                
                if ((int)activity.get(0)[4]==5) {
                    outputValues = "0" + csvSplitBy + "0" + csvSplitBy + "0" + csvSplitBy + "0" + csvSplitBy + "0" + csvSplitBy + "1" + csvSplitBy + "0";
                }
                if ((int)activity.get(0)[4]==6) {
                    outputValues = "0" + csvSplitBy + "0" + csvSplitBy + "0" + csvSplitBy + "0" + csvSplitBy + "0" + csvSplitBy + "0" + csvSplitBy + "1";
                }
                
                //insert into dataset
                pw.print(mean(oneActivityXAxisValues) + csvSplitBy + mean(oneActivityYAxisValues) + csvSplitBy + mean(oneActivityZAxisValues) + csvSplitBy
                        + standardDeviation(oneActivityXAxisValues) + csvSplitBy + standardDeviation(oneActivityYAxisValues) + csvSplitBy + standardDeviation(oneActivityZAxisValues) + csvSplitBy
                        + variance(oneActivityXAxisValues) + csvSplitBy + variance(oneActivityYAxisValues) + csvSplitBy + variance(oneActivityZAxisValues) + csvSplitBy
                        + meanAbsoluteDeviation(oneActivityXAxisValues) + csvSplitBy + meanAbsoluteDeviation(oneActivityYAxisValues) + csvSplitBy + meanAbsoluteDeviation(oneActivityZAxisValues) + csvSplitBy
                        + rootMeanSquare(oneActivityXAxisValues) + csvSplitBy + rootMeanSquare(oneActivityYAxisValues) + csvSplitBy + rootMeanSquare(oneActivityZAxisValues) + csvSplitBy
                        + interquartileRange(oneActivityXAxisValues) + csvSplitBy + interquartileRange(oneActivityYAxisValues) + csvSplitBy + interquartileRange(oneActivityZAxisValues) + csvSplitBy
                        + energy(oneActivityXAxisValues) + csvSplitBy + energy(oneActivityYAxisValues) + csvSplitBy + energy(oneActivityZAxisValues) + csvSplitBy
                        + correlation(oneActivityXAxisValues, oneActivityYAxisValues) + csvSplitBy + correlation(oneActivityYAxisValues, oneActivityZAxisValues) + csvSplitBy + correlation(oneActivityZAxisValues, oneActivityXAxisValues) + csvSplitBy
                        + outputValues);
                pw.println();
                
                numberOfFramesComputed++;
                
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
        
        System.out.println("pocet frameov: " + numberOfFramesComputed);
    }
    
    public static void processData(String csvFile){
        String line = "";
        String cvsSplitBy = ",";
        String currentActivity;
        String activityCode = "";
        try (BufferedReader br = new BufferedReader(new FileReader(csvFile))) {
            br.readLine();//preskoc prvy riadok s nazvami stlpcov
            while ((line = br.readLine()) != null) {
                String[] zaznam = line.split(cvsSplitBy);
                
                currentActivity = zaznam[9];
                
                switch(currentActivity){
                    case "stand":
                        activityCode = "0";
                        break;
                    case "walk":
                        activityCode = "1";
                        break;
                    case "stairsup":
                        activityCode = "2";
                        break;
                    case "stairsdown":
                        activityCode = "3";
                        break;
                    case "sit":
                        activityCode = "4";
                        break;
                    default:
                        //aktivita ktoru neuvazujeme
                        activityCode = "10";
                }
                
                if (activityCode.equals("10")) {
                    continue;
                }
                   
                oneActivityTimes = new ArrayList<>();
                oneActivity = new ArrayList<>();
                //add values xAxis, yAxis, zAxis, activityCode
                oneActivity.add(new double[]{Double.parseDouble(zaznam[3]), Double.parseDouble(zaznam[4]),
                                Double.parseDouble(zaznam[5]), Double.parseDouble(activityCode)});
                oneActivityTimes.add(new BigInteger(zaznam[2]));
                while ((line = br.readLine()) != null) {
                    zaznam = line.split(cvsSplitBy);
                    if (zaznam[9].equals(currentActivity)) {                          
                        //add values for xAxis, yAxis, zAxis, activityCode
                        oneActivity.add(new double[]{Double.parseDouble(zaznam[3]), Double.parseDouble(zaznam[4]),
                                Double.parseDouble(zaznam[5]), Double.parseDouble(activityCode)});
                        oneActivityTimes.add(new BigInteger(zaznam[2]));                                                   
                    } else {
                        //tu spracovat data aktivity a dat do suboru
                        System.out.println("pocet zaznamov aktivity " + currentActivity + "  :" + oneActivity.size() + " dlzka casov: " + oneActivityTimes.size());
                            
                        //len na histogramy
                        //if (currentActivity.equals("walk")) {
                            compute(oneActivity, oneActivityTimes); 
                        //}
                            
                        oneActivity = new ArrayList<>();
                        oneActivityTimes = new ArrayList<>();
                        break;
                    }
                } 
                //ak sme došli na koniec suboru a posledna aktivita
                if (oneActivity.size() != 0) {
                    System.out.println("pocet zaznamov aktivity " + currentActivity + "  :" + oneActivity.size());
                    
                    compute(oneActivity, oneActivityTimes);                     
                    
                    oneActivity = new ArrayList<>();
                    oneActivityTimes = new ArrayList<>();
                }
            }
        } catch (IOException e) {
            e.printStackTrace();
        }  
    }

    //here we should divide activity into windows
    //and compute our features for every window and insert one line with feature values into dataset file
    private static void compute(List<double[]> activity, List<BigInteger> activityTimes) {
        celkovyCas += activityTimes.get(activityTimes.size()-1).subtract(activityTimes.get(0)).doubleValue()/1000000000;
        int numberOfFramesComputed = 0;
        PrintWriter pw = null;
        //int numberOfSamples = activity.size();
        BigInteger startTime = activityTimes.get(0); 
        BigInteger endTime = activityTimes.get(activityTimes.size()-1);
        int currentStartListPosition = 0;

        if (endTime.subtract(startTime).compareTo(nanosecondsInFrame) < 0) {
            return;
        }
        
        BigInteger frameEndTime = startTime.add(nanosecondsInFrame);
        
        try {
            pw = new PrintWriter(new FileOutputStream(new File(processedDataFile),true /*append*/));
            while (frameEndTime.compareTo(endTime) <= 0) {
                //here we compute window containing samples from between startTime and frameEndTime
                //compute features
                
                //get oneActivityAxisValues for current frame
                int i = currentStartListPosition;
                while (oneActivityTimes.get(i).compareTo(frameEndTime)<0) {
                    i++;                    
                }
                
                oneActivityXAxisValues = new double[i-currentStartListPosition];
                oneActivityYAxisValues = new double[i-currentStartListPosition];
                oneActivityZAxisValues = new double[i-currentStartListPosition];
                
                
                for (int j = 0; j < oneActivityXAxisValues.length; j++) {
                    oneActivityXAxisValues[j]=oneActivity.get(currentStartListPosition+j)[0];
                    oneActivityYAxisValues[j]=oneActivity.get(currentStartListPosition+j)[1];
                    oneActivityZAxisValues[j]=oneActivity.get(currentStartListPosition+j)[2];
                }          
                
                String outputValues = "";
                
                if ((int)activity.get(0)[3]==0) {
                        outputValues = "1" + csvSplitBy + "0" + csvSplitBy + "0" + csvSplitBy + "0" + csvSplitBy + "0" + csvSplitBy + "0" + csvSplitBy + "0";
                    } 
                    if ((int)activity.get(0)[3]==1) {
                        outputValues = "0" + csvSplitBy + "1" + csvSplitBy + "0" + csvSplitBy + "0" + csvSplitBy + "0" + csvSplitBy + "0" + csvSplitBy + "0";
                    }
                    if ((int)activity.get(0)[3]==2) {
                        outputValues = "0" + csvSplitBy + "0" + csvSplitBy + "1" + csvSplitBy + "0" + csvSplitBy + "0" + csvSplitBy + "0" + csvSplitBy + "0";
                    }
                    if ((int)activity.get(0)[3]==3) {
                        outputValues = "0" + csvSplitBy + "0" + csvSplitBy + "0" + csvSplitBy + "1" + csvSplitBy + "0" + csvSplitBy + "0" + csvSplitBy + "0";
                    }
                    if ((int)activity.get(0)[3]==4) {
                        outputValues = "0" + csvSplitBy + "0" + csvSplitBy + "0" + csvSplitBy + "0" + csvSplitBy + "1" + csvSplitBy + "0" + csvSplitBy + "0";
                    }
                
                //insert into dataset
                pw.print(mean(oneActivityXAxisValues) + csvSplitBy + mean(oneActivityYAxisValues) + csvSplitBy + mean(oneActivityZAxisValues) + csvSplitBy
                        + standardDeviation(oneActivityXAxisValues) + csvSplitBy + standardDeviation(oneActivityYAxisValues) + csvSplitBy + standardDeviation(oneActivityZAxisValues) + csvSplitBy
                        + variance(oneActivityXAxisValues) + csvSplitBy + variance(oneActivityYAxisValues) + csvSplitBy + variance(oneActivityZAxisValues) + csvSplitBy
                        + meanAbsoluteDeviation(oneActivityXAxisValues) + csvSplitBy + meanAbsoluteDeviation(oneActivityYAxisValues) + csvSplitBy + meanAbsoluteDeviation(oneActivityZAxisValues) + csvSplitBy
                        + rootMeanSquare(oneActivityXAxisValues) + csvSplitBy + rootMeanSquare(oneActivityYAxisValues) + csvSplitBy + rootMeanSquare(oneActivityZAxisValues) + csvSplitBy
                        + interquartileRange(oneActivityXAxisValues) + csvSplitBy + interquartileRange(oneActivityYAxisValues) + csvSplitBy + interquartileRange(oneActivityZAxisValues) + csvSplitBy
                        + energy(oneActivityXAxisValues) + csvSplitBy + energy(oneActivityYAxisValues) + csvSplitBy + energy(oneActivityZAxisValues) + csvSplitBy
                        + correlation(oneActivityXAxisValues, oneActivityYAxisValues) + csvSplitBy + correlation(oneActivityYAxisValues, oneActivityZAxisValues) + csvSplitBy + correlation(oneActivityZAxisValues, oneActivityXAxisValues) + csvSplitBy
                        + outputValues);
                pw.println();
                
                numberOfFramesComputed++;
                
                //update variables currentStartListPosition,startTime,frameEndTime
                BigInteger newFrameStartTime = startTime.add(nanosecondsInFrame.divide(new BigInteger("2"))); //diveded by 2 because we have 50% window overlapping
                while (oneActivityTimes.get(currentStartListPosition).compareTo(newFrameStartTime)< 0) {                    
                    currentStartListPosition++;
                }
                startTime = oneActivityTimes.get(currentStartListPosition);
                frameEndTime = startTime.add(nanosecondsInFrame); 
            }         
        } catch (Exception e) {
            e.printStackTrace();
        } finally {
            if(pw != null){
                pw.close();
            }
        }
        
        System.out.println("pocet frameov: " + numberOfFramesComputed);
    }
    
    //metody na vypocty featurov, ktore sa budu volat v compute metode
    
    //simple mean
    private static double mean(double[] data){
        mean = new Mean().evaluate(data);
        return mean;
    }
    
    //standard deviation
    private static double standardDeviation(double[] data){
        return new StandardDeviation().evaluate(data, mean);
    }
    
    //variance
    private static double variance(double[] data){
        return new Variance().evaluate(data, mean);
    }
    
    //mean absolute deviation
    private static double meanAbsoluteDeviation(double[] data){
        double sum = 0.0;
        for (int i = 0; i < data.length; i++) {
            sum = sum + Math.abs(data[i] - mean);
        }
        return sum/data.length;
    }
    
    //root mean square
    private static double rootMeanSquare(double[] data){
        double sum = 0.0;
        for (int i = 0; i < data.length; i++) {
            sum = sum + Math.pow(data[i],2);
        }
        return Math.sqrt(sum/data.length);
    }
    
    //interquartile range
    private static double interquartileRange(double[] data){
        DescriptiveStatistics ds = new DescriptiveStatistics(data);
        return ds.getPercentile(75) - ds.getPercentile(25);
    }
    
    //energy
    private static double energy(double[] data){
        FastFourierTransformer fft = new FastFourierTransformer(DftNormalization.STANDARD);
        
        //tu este uprava dlzky dat na mocninu 2
        int mocnina = 1;
        while (mocnina<=data.length) {
            mocnina = mocnina*2;            
        }
        double[] newArray = Arrays.copyOf(data,mocnina);
                
        Complex[] newData = fft.transform(newArray, TransformType.INVERSE);
        double sum = 0;
        for (int i = 0; i < newData.length; i++) {
            sum = sum + Math.pow(newData[i].getReal(),2);            
        }
        return sum/newData.length;
    }
    
    //correlation between axes
    private static double correlation(double[] firstArray, double[] secondArray){
        Covariance cov = new Covariance();
        return cov.covariance(firstArray, secondArray)/(standardDeviation(firstArray)*standardDeviation(secondArray));
    }
    
    
    /*
        NEURAL NETWORK
    */
    
    public void run() {
        System.out.println("Creating training and test set from file...");
        int inputsCount = 24;
        int outputsCount = 7;
        
        //Create data set from file
        DataSet dataSet = DataSet.createFromFile(processedDataFile, inputsCount, outputsCount, csvSplitBy);
        dataSet.shuffle();

        //Normalizing data set
        Normalizer normalizer = new MaxNormalizer();
        normalizer.normalize(dataSet);
        
        //save normalized dataset to file 
        //dataSet.saveAsTxt("dataset.csv", csvSplitBy);

        //Creating training set (70%) and test set (30%)
        List<DataSet> trainingAndTestSet = dataSet.split(70, 30);
        DataSet trainingSet = trainingAndTestSet.get(0);
        DataSet testSet = trainingAndTestSet.get(1);

        //Create MultiLayerPerceptron neural network
        MultiLayerPerceptron neuralNet = new MultiLayerPerceptron(inputsCount, 30, outputsCount);

        //attach listener to learning rule
        MomentumBackpropagation learningRule = (MomentumBackpropagation) neuralNet.getLearningRule();
        learningRule.addListener(this);
        
        learningRule.setLearningRate(0.03);
        learningRule.setMaxError(0.001);
        learningRule.setMaxIterations(5000);

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
            int predicted = maxOutput(networkOutput);

            //Finding actual output
            double[] networkDesiredOutput = testSetRow.getDesiredOutput();
            int ideal = maxOutput(networkDesiredOutput);
            
            //Colecting data for network evaluation
            keepScore(predicted, ideal);
        }
        
        System.out.println("Total cases: " + this.count[7] + ". ");
        System.out.println("Correctly predicted cases: " + this.correct[7] + ". ");
        System.out.println("Incorrectly predicted cases: " + (this.count[7] - this.correct[7] - unpredicted) + ". ");
        System.out.println("Unrecognized cases: " + unpredicted + ". ");
        double percentTotal = (double) this.correct[7] * 100 / (double) this.count[7];
        System.out.println("Predicted correctly: " + formatDecimalNumber(percentTotal) + "%. ");

        for (int i = 0; i < correct.length - 1; i++) {
            if (this.count[i]==0) {
                continue;
            }
            double p = (double) this.correct[i] * 100.0 / (double) this.count[i];
            System.out.println("Segment class: " + getClasificationClass(i) + " - Correct/total: "
                    + this.correct[i] + "/" + count[i] + "(" + formatDecimalNumber(p) + "%). ");
        }
        
        System.out.println("Results matrix:");
        System.out.println("    standing    walking walkingUpstairs walkingDownstairs   sit  elevatorUp elevatorDown");
        for (int i = 0; i < 7; i++) {
            System.out.println(getClasificationClass(i) + " " + testingResults[i][0] + " " + testingResults[i][1] + " " + testingResults[i][2]
            + " " + testingResults[i][3] + " " + testingResults[i][4] + " " + testingResults[i][5] + " " + testingResults[i][6]);
        }
        
        
        this.count = new int[8];
        this.correct = new int[8];
        unpredicted = 0;
        this.testingResults = new int[7][7];
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

    //Metod determines the maximum output. Maximum output is network prediction for one row.
    public static int maxOutput(double[] array) {
        double max = array[0];
        int index = 0;
        
        for (int i = 0; i < array.length; i++) {
            if (array[i] > max) {
                index = i;
                max = array[i];
            }
        }
        //If maximum is less than 0.5, that prediction will not count. 
        //if (max < 0.5) {
        //    return -1;
        //}
        return index;
    }

    //Colecting data to evaluate network.
    public void keepScore(int prediction, int ideal) {
        count[ideal]++;
        count[7]++;

        if (prediction == ideal) {
            correct[ideal]++;
            correct[7]++;
        }
        if (prediction == -1) {
            unpredicted++;
        }
        
        testingResults[ideal][prediction]++;
    }
    
    //Formating decimal number to have 3 decimal places
    public String formatDecimalNumber(double number) {
        return new BigDecimal(number).setScale(4, RoundingMode.HALF_UP).toString();
    }
    
    public String getClasificationClass(int i) {
        switch (i) {
            case 0:
                return "standing";
            case 1:
                return "walking";
            case 2:
                return "walkingUpstairs";
            case 3:
                return "walkingDownstairs";
            case 4:
                return "sit";
            case 5:
                return "elevatorUp"; 
            case 6:
                return "elevatorDown"; 
            default:
                return "error";
        }
    }
}
