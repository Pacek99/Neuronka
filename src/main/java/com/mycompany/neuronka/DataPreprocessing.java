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
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Set;
import org.neuroph.core.*;
import org.neuroph.core.data.DataSet;
import org.neuroph.core.data.DataSetRow;
import org.neuroph.nnet.*;

/**
 *
 * @author Patrik
 */
public class DataPreprocessing {
    
    private static Map<Double, Double> map;
    private static String sensor = "AK09918C";
    private static String processedDataFile = "processedData.csv";
    private static List<double[]> oneActivity;
    private static int numberOfSamplesInFrame = 600;
    private static String csvSplitBy = "\t";
    
    public static void main(String csvFile, String activity){    
        String line = "";
        
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
        
        //processData(sensor, "C:/Users/Patrik/Desktop/Bakalarka - SensorRecorder dáta/P. Rojek/indora-1540484172540.csv");
        
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
                    //oneActivity.add(new double[]{Double.valueOf(zaznam[0]),Double.parseDouble(zaznam[3]),Double.parseDouble(zaznam[4]),
                    //    Double.parseDouble(zaznam[5]),Double.parseDouble(activityCode)});
                    oneActivity.add(new double[]{Double.valueOf(zaznam[0]), Math.sqrt(Double.parseDouble(zaznam[3])*Double.parseDouble(zaznam[3])
                                        + Double.parseDouble(zaznam[4])*Double.parseDouble(zaznam[4])
                                        + Double.parseDouble(zaznam[5])*Double.parseDouble(zaznam[5])),Double.parseDouble(activityCode)});
                    while ((line = br.readLine()) != null) {
                        zaznam = line.split(cvsSplitBy);
                        if (zaznam[1].equals(currentActivity)) {
                            if (zaznam[2].equals(sensor)) {
                                //3 axes separate
                                //oneActivity.add(new double[]{Double.valueOf(zaznam[0]),Double.parseDouble(zaznam[3]),Double.parseDouble(zaznam[4]),
                                //Double.parseDouble(zaznam[5]),Double.parseDouble(activityCode)});
                                
                                //(Math.sqrt(x*x+y*y+z*z)) 
                                oneActivity.add(new double[]{Double.valueOf(zaznam[0]), Math.sqrt(Double.parseDouble(zaznam[3])*Double.parseDouble(zaznam[3])
                                        + Double.parseDouble(zaznam[4])*Double.parseDouble(zaznam[4])
                                        + Double.parseDouble(zaznam[5])*Double.parseDouble(zaznam[5])),Double.parseDouble(activityCode)});
                            }                            
                        } else {
                            //TODO tu spreacovat data aktivity a dat do suboru
                            System.out.println("pocet zaznamov aktivity " + currentActivity + "  :" + oneActivity.size());
                            compute(oneActivity);
                            break;
                        }
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
        int numberOfSamples = activity.size();
        int currentPosition = 0;
        if (numberOfSamples<numberOfSamplesInFrame) {
            return;
        }
        try {
            pw = new PrintWriter(new FileOutputStream(new File(processedDataFile),true /*append*/));
            while (currentPosition + numberOfSamplesInFrame <= numberOfSamples) {
                //here we compute window containing samples from currentPosition to currentPosition + numberOfSamplesInFrame
                //compute features
                
                //insert into dataset
                pw.print(mean(currentPosition) + csvSplitBy + standardDeviation(currentPosition) + csvSplitBy + (int)activity.get(0)[2]);
                pw.println();
                
                currentPosition += numberOfSamplesInFrame/2; //diveded by 2 because we have 50% window overlapping
            }
            
            /*
            //only for testing for now
            for(double[] line: oneActivity){
                //pw.print(line[0] + cvsSplitBy + line[1] + cvsSplitBy + line[2] + cvsSplitBy + line[3] + cvsSplitBy + (int)line[4]);
                pw.print(line[0] + cvsSplitBy + line[1] + cvsSplitBy + (int)line[2]);
                pw.println();
            }     
            */          
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
    private static double mean(int fromIndex){
        double sum = 0.0;
        for (int i = fromIndex; i < fromIndex + numberOfSamplesInFrame; i++) {
            sum = sum + oneActivity.get(i)[1];
        }
        return sum/numberOfSamplesInFrame;
    }
    
    //standard deviation
    private static double standardDeviation(int fromIndex){
        double mean = mean(fromIndex);
        double sum = 0.0;
        for (int i = fromIndex; i < fromIndex + numberOfSamplesInFrame; i++) {
            sum = sum + Math.pow(oneActivity.get(i)[1] - mean, 2);
        }
        return Math.sqrt(sum/(numberOfSamplesInFrame-1));                
    }
    
     /**
     * Prints network output for the each element from the specified training set. 
     * @param neuralNet neural network 
     * @param trainingSet training set 
     */ 
    public static void testNeuralNetwork(NeuralNetwork neuralNet) { 
 
        neuralNet.setInput(new double[]{43.18602115267101, 3.2018294701549244}); 
        neuralNet.calculate(); 
        double[] networkOutput = neuralNet.getOutput(); 
 
        System.out.println(" Output: " + Arrays.toString( networkOutput) ); 
         
    } 
}
