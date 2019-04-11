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
public class DataPreprocessing implements LearningEventListener{
    
    private static Map<Double, Double> map;
    private static String sensor = "AK09918C";
    private static String processedDataFile = "processedData.csv";
    private static List<double[]> oneActivity;
    private static double[] oneActivityAxisValues;
    
    private static double[] oneActivityXAxisValues;
    private static double[] oneActivityYAxisValues;
    private static double[] oneActivityZAxisValues;
    
    private static int milisecondsInFrame = 3000;
    private static int numberOfSamplesInFrame = 600;
    private static String csvSplitBy = "\t";
    
    //variables for feature calc
    private static double mean;
       
   // if output is greater then this value it is considered as walking
    float classificationThreshold = 0.5f;
    
    //Important for evaluating network result
    public int[] count = new int[7];
    public int[] correct = new int[7];
    int unpredicted = 0;

    public static void main(String[] args){    
        Map<String, String> subory = new HashMap<String,String>();
        
        // priečinok B. Taylorová
        subory.put("src/main/resources/B. Taylorová/indora-1549482603748.csv", "MPU6500 Acceleration Sensor");        
        subory.put("src/main/resources/B. Taylorová/indora-1549482689471.csv", "MPU6500 Acceleration Sensor");        
        subory.put("src/main/resources/B. Taylorová/indora-1549482748787.csv", "MPU6500 Acceleration Sensor");        
        subory.put("src/main/resources/B. Taylorová/indora-1549482827151.csv", "MPU6500 Acceleration Sensor");        
        subory.put("src/main/resources/B. Taylorová/indora-1549482928447.csv", "MPU6500 Acceleration Sensor");        
        subory.put("src/main/resources/B. Taylorová/indora-1549482960030.csv", "MPU6500 Acceleration Sensor");
                      
        //subory.put("src/main/resources/B. Taylorová/indora-1552906231896.csv", "MPU6500 Acceleration Sensor");       
        //subory.put("src/main/resources/B. Taylorová/indora-1552906271137.csv", "MPU6500 Acceleration Sensor");       
        //subory.put("src/main/resources/B. Taylorová/indora-1552906425702.csv", "MPU6500 Acceleration Sensor");       
        //subory.put("src/main/resources/B. Taylorová/indora-1552906472467.csv", "MPU6500 Acceleration Sensor");
               
        //subory.put("src/main/resources/B. Taylorová/indora-1552906553847.csv", "MPU6500 Acceleration Sensor");       
        //subory.put("src/main/resources/B. Taylorová/indora-1553278125310.csv", "MPU6500 Acceleration Sensor");       
        //subory.put("src/main/resources/B. Taylorová/indora-1553278129057.csv", "MPU6500 Acceleration Sensor");       
        //subory.put("src/main/resources/B. Taylorová/indora-1553278160954.csv", "MPU6500 Acceleration Sensor");  
        
        // priečinok M. Sochuliak
        subory.put("src/main/resources/M. Sochuliak/indora-1549012172677.csv", "ACCELEROMETER");        
        subory.put("src/main/resources/M. Sochuliak/indora-1549021777198.csv", "ACCELEROMETER");        
        subory.put("src/main/resources/M. Sochuliak/indora-1549022025135.csv", "ACCELEROMETER");        
        subory.put("src/main/resources/M. Sochuliak/indora-1549022068213.csv", "ACCELEROMETER");
        
        // priečinok P. Kendra
        subory.put("src/main/resources/P. Kendra/indora-1549541475108.csv", sensor);        
        subory.put("src/main/resources/P. Kendra/indora-1549541522778.csv", sensor);        
        subory.put("src/main/resources/P. Kendra/indora-1549541559653.csv", sensor);        
        subory.put("src/main/resources/P. Kendra/indora-1549541572516.csv", sensor);        
        subory.put("src/main/resources/P. Kendra/indora-1549541582979.csv", sensor);        
        subory.put("src/main/resources/P. Kendra/indora-1549541633702.csv", sensor);        
        subory.put("src/main/resources/P. Kendra/indora-1549541647313.csv", sensor);        
        subory.put("src/main/resources/P. Kendra/indora-1549541673057.csv", sensor);        
        subory.put("src/main/resources/P. Kendra/indora-1549541682687.csv", sensor);        
        subory.put("src/main/resources/P. Kendra/indora-1549541695967.csv", sensor);        
        subory.put("src/main/resources/P. Kendra/indora-1549541708344.csv", sensor);
        
        // priečinok P. Rojek
        subory.put("src/main/resources/P. Rojek/indora-1540484172540.csv", sensor);        
        subory.put("src/main/resources/P. Rojek/indora-1540484443308.csv", sensor);        
        subory.put("src/main/resources/P. Rojek/indora-1540484680716.csv", sensor);        
        subory.put("src/main/resources/P. Rojek/indora-1540546792759.csv", sensor);        
        subory.put("src/main/resources/P. Rojek/indora-1540554936805.csv", sensor);
        subory.put("src/main/resources/P. Rojek/indora-1554699869876.csv", sensor);        
        subory.put("src/main/resources/P. Rojek/indora-1554699898743.csv", sensor);        
        subory.put("src/main/resources/P. Rojek/indora-1554699925682.csv", sensor);        
        subory.put("src/main/resources/P. Rojek/indora-1554699994623.csv", sensor);        
        subory.put("src/main/resources/P. Rojek/indora-1554710761298.csv", sensor);        
        subory.put("src/main/resources/P. Rojek/indora-1554728381607.csv", sensor);        
        subory.put("src/main/resources/P. Rojek/indora-1554728410434.csv", sensor);
        
        // priečinok Š. Rojek
        subory.put("src/main/resources/Š. Rojek/indora-1540362934669.csv", sensor);        
        subory.put("src/main/resources/Š. Rojek/indora-1540363171233.csv", sensor);        
        subory.put("src/main/resources/Š. Rojek/indora-1540363247042.csv", sensor);        
        subory.put("src/main/resources/Š. Rojek/indora-1540363314900.csv", sensor);        
        subory.put("src/main/resources/Š. Rojek/indora-1540363406576.csv", sensor);
        
        // priečinok sk.upjs.indora.sensorsrecorder         
        subory.put("src/main/resources/sk.upjs.indora.sensorsrecorder/indora-1549541475108.csv", sensor);
        subory.put("src/main/resources/sk.upjs.indora.sensorsrecorder/indora-1549541522778.csv", sensor);
        subory.put("src/main/resources/sk.upjs.indora.sensorsrecorder/indora-1549541552672.csv", sensor);        
        subory.put("src/main/resources/sk.upjs.indora.sensorsrecorder/indora-1549541559653.csv", sensor);        
        subory.put("src/main/resources/sk.upjs.indora.sensorsrecorder/indora-1549541572516.csv", sensor); 
        
        subory.put("src/main/resources/sk.upjs.indora.sensorsrecorder/indora-1549541582979.csv", sensor);
        subory.put("src/main/resources/sk.upjs.indora.sensorsrecorder/indora-1549541633702.csv", sensor);
        subory.put("src/main/resources/sk.upjs.indora.sensorsrecorder/indora-1549541647313.csv", sensor);        
        subory.put("src/main/resources/sk.upjs.indora.sensorsrecorder/indora-1549541673057.csv", sensor);        
        subory.put("src/main/resources/sk.upjs.indora.sensorsrecorder/indora-1549541682687.csv", sensor); 
        
        subory.put("src/main/resources/sk.upjs.indora.sensorsrecorder/indora-1549541695967.csv", sensor);
        subory.put("src/main/resources/sk.upjs.indora.sensorsrecorder/indora-1549541708344.csv", sensor);
        subory.put("src/main/resources/sk.upjs.indora.sensorsrecorder/indora-1553014481636.csv", sensor);        
        subory.put("src/main/resources/sk.upjs.indora.sensorsrecorder/indora-1553014543983.csv", sensor);        
        subory.put("src/main/resources/sk.upjs.indora.sensorsrecorder/indora-1553014570035.csv", sensor); 
        
        subory.put("src/main/resources/sk.upjs.indora.sensorsrecorder/indora-1553019600096.csv", sensor);
        subory.put("src/main/resources/sk.upjs.indora.sensorsrecorder/indora-1553019658511.csv", sensor);
        subory.put("src/main/resources/sk.upjs.indora.sensorsrecorder/indora-1553019699799.csv", sensor);        
        subory.put("src/main/resources/sk.upjs.indora.sensorsrecorder/indora-1553589053903.csv", sensor);        
        subory.put("src/main/resources/sk.upjs.indora.sensorsrecorder/indora-1553589171764.csv", sensor); 
        
        subory.put("src/main/resources/sk.upjs.indora.sensorsrecorder/indora-1553589248068.csv", sensor);
        subory.put("src/main/resources/sk.upjs.indora.sensorsrecorder/indora-1553589338226.csv", sensor);
        subory.put("src/main/resources/sk.upjs.indora.sensorsrecorder/indora-1553589368041.csv", sensor);        
        subory.put("src/main/resources/sk.upjs.indora.sensorsrecorder/indora-1553589528999.csv", sensor);        
        subory.put("src/main/resources/sk.upjs.indora.sensorsrecorder/indora-1553609125909.csv", sensor); 
        
        subory.put("src/main/resources/sk.upjs.indora.sensorsrecorder/indora-1553609140518.csv", sensor);
        subory.put("src/main/resources/sk.upjs.indora.sensorsrecorder/indora-1553609169349.csv", sensor);
        subory.put("src/main/resources/sk.upjs.indora.sensorsrecorder/indora-1553609197288.csv", sensor);        
        subory.put("src/main/resources/sk.upjs.indora.sensorsrecorder/indora-1553609224970.csv", sensor);        
        subory.put("src/main/resources/sk.upjs.indora.sensorsrecorder/indora-1553609253485.csv", sensor); 
        
        subory.put("src/main/resources/sk.upjs.indora.sensorsrecorder/indora-1553610218773.csv", sensor);
        subory.put("src/main/resources/sk.upjs.indora.sensorsrecorder/indora-1553610244950.csv", sensor);
        subory.put("src/main/resources/sk.upjs.indora.sensorsrecorder/indora-1553610273270.csv", sensor);        
        subory.put("src/main/resources/sk.upjs.indora.sensorsrecorder/indora-1553610302262.csv", sensor);        
        subory.put("src/main/resources/sk.upjs.indora.sensorsrecorder/indora-1553610326669.csv", sensor); 
        
        // vygenerovanie datasetu 
        for (Map.Entry<String, String> entry : subory.entrySet()) {
            String key = entry.getKey();
            String value = entry.getValue();
            System.out.println("Subor: " + key + " a akcelerometer: " + value);
            processData(value,key);
        }    
        
        
        /* Javaplot
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
        //(new DataPreprocessing()).run();
    }
    
    /*
    //toto patri k Javaplotu
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
    */
    
    public static void processData(String sensor, String csvFile){
        String line = "";
	String cvsSplitBy = "\t";
        String currentActivity;
        String activityCode = "";
        try (BufferedReader br = new BufferedReader(new FileReader(csvFile))) {
            while ((line = br.readLine()) != null) {
                String[] zaznam = line.split(cvsSplitBy);
                currentActivity = zaznam[1];
                
                if (zaznam[2].equals(sensor)) {                     
                    if (currentActivity.equals("standing")) {
                        activityCode = "0";
                    } 
                    if (currentActivity.equals("walking")) {
                        activityCode = "1";
                    }
                    if (currentActivity.equals("walkingUpstairs")) {
                        activityCode = "2";
                    }
                    if (currentActivity.equals("walkingDownstairs")) {
                        activityCode = "3";
                    }
                    if (currentActivity.equals("elevatorUp")) {
                        activityCode = "4";
                    }
                    if (currentActivity.equals("elevatorDown")) {
                        activityCode = "5";
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
                
                if ((int)activity.get(0)[4]==0) {
                        outputValues = "1" + csvSplitBy + "0" + csvSplitBy + "0" + csvSplitBy + "0" + csvSplitBy + "0" + csvSplitBy + "0";
                    } 
                    if ((int)activity.get(0)[4]==1) {
                        outputValues = "0" + csvSplitBy + "1" + csvSplitBy + "0" + csvSplitBy + "0" + csvSplitBy + "0" + csvSplitBy + "0";
                    }
                    if ((int)activity.get(0)[4]==2) {
                        outputValues = "0" + csvSplitBy + "0" + csvSplitBy + "1" + csvSplitBy + "0" + csvSplitBy + "0" + csvSplitBy + "0";
                    }
                    if ((int)activity.get(0)[4]==3) {
                        outputValues = "0" + csvSplitBy + "0" + csvSplitBy + "0" + csvSplitBy + "1" + csvSplitBy + "0" + csvSplitBy + "0";
                    }
                    if ((int)activity.get(0)[4]==4) {
                        outputValues = "0" + csvSplitBy + "0" + csvSplitBy + "0" + csvSplitBy + "0" + csvSplitBy + "1" + csvSplitBy + "0";
                    }
                    if ((int)activity.get(0)[4]==5) {
                        outputValues = "0" + csvSplitBy + "0" + csvSplitBy + "0" + csvSplitBy + "0" + csvSplitBy + "0" + csvSplitBy + "1";
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
        int outputsCount = 6;
        
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
        
        learningRule.setLearningRate(0.01);
        learningRule.setMaxError(0.001);
        learningRule.setMaxIterations(10000);

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
        
        System.out.println("Total cases: " + this.count[6] + ". ");
        System.out.println("Correctly predicted cases: " + this.correct[6] + ". ");
        System.out.println("Incorrectly predicted cases: " + (this.count[6] - this.correct[6] - unpredicted) + ". ");
        System.out.println("Unrecognized cases: " + unpredicted + ". ");
        double percentTotal = (double) this.correct[6] * 100 / (double) this.count[6];
        System.out.println("Predicted correctly: " + formatDecimalNumber(percentTotal) + "%. ");

        for (int i = 0; i < correct.length - 1; i++) {
            double p = (double) this.correct[i] * 100.0 / (double) this.count[i];
            System.out.println("Segment class: " + getClasificationClass(i) + " - Correct/total: "
                    + this.correct[i] + "/" + count[i] + "(" + formatDecimalNumber(p) + "%). ");
        }
        
        this.count = new int[7];
        this.correct = new int[7];
        unpredicted = 0;
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
        if (max < 0.5) {
            return -1;
        }
        return index;
    }

    //Colecting data to evaluate network.
    public void keepScore(int prediction, int ideal) {
        count[ideal]++;
        count[6]++;

        if (prediction == ideal) {
            correct[ideal]++;
            correct[6]++;
        }
        if (prediction == -1) {
            unpredicted++;
        }
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
                return "elevatorUp";
            case 5:
                return "elevatorDown";          
            default:
                return "error";
        }
    }
}
