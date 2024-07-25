/*
 * Copyright (c) 2019-2021 Qualcomm Technologies, Inc.
 * All Rights Reserved.
 * Confidential and Proprietary - Qualcomm Technologies, Inc.
 */
package com.qualcomm.qti.psnpedemo.processor;

import android.util.Log;

import com.qualcomm.qti.psnpedemo.components.BenchmarkApplication;
import com.qualcomm.qti.psnpedemo.networkEvaluation.ClassificationResult;
import com.qualcomm.qti.psnpedemo.networkEvaluation.EvaluationCallBacks;
import com.qualcomm.qti.psnpedemo.networkEvaluation.ModelInfo;
import com.qualcomm.qti.psnpedemo.networkEvaluation.Result;
import com.qualcomm.qti.psnpe.PSNPEManager;
import com.qualcomm.qti.psnpedemo.utils.Util;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

public class ClassificationPostprocessor extends PostProcessor{
    private String modelLabelPath;
    private String groundTruthPath;
    private int top1Count;
    private int top5Count;
    private int totNum;

    public  ClassificationPostprocessor(EvaluationCallBacks evaluationCallBacks, ModelInfo modelInfo, int imageNumber)  {
        super(imageNumber);
        String modelName = modelInfo.getModelName();
        String dataSetName = modelInfo.getDataSetName();
        String packagePath = BenchmarkApplication.getCustomApplicationContext().getExternalFilesDir("").getAbsolutePath();
        groundTruthPath = packagePath + "/datasets/" + dataSetName + "/labels.txt";

        if(modelName.contains("inception")) {
            modelLabelPath = packagePath + "/datasets/" + dataSetName + "/labels_dlc_inceptionv3.txt";
        } else if(modelName.contains("mobilenet")) {
            modelLabelPath = packagePath + "/datasets/" + dataSetName + "/ilsvrc_2012_labels.txt";
        } else {
            modelLabelPath = packagePath + "/datasets/" + dataSetName + "/imagenet_slim_labels.txt";
        }
        this.top1Count = 0;
        this.top5Count = 0;
        this.totNum = 0;
        this.evaluationCallBacks = evaluationCallBacks;
    }

    private String[] loadModelLabels() {
        InputStream modelLabelsStream = null;
        try {
            modelLabelsStream = new FileInputStream(modelLabelPath);
        } catch (FileNotFoundException e) {
            Log.d(TAG, "modelLabelPath: " + modelLabelPath);
            e.printStackTrace();
            evaluationCallBacks.showErrorText("modelLabelPath not exit: " + modelLabelPath);
            return new String[0];
        }
        List<String> list = new LinkedList<>();
        BufferedReader inputStream = new BufferedReader(new InputStreamReader(modelLabelsStream));
        String line;
        while (true) {
            try {
                if (!((line = inputStream.readLine()) != null)) break;
            } catch (IOException e) {
                e.printStackTrace();
                continue;
            }
            list.add(line);
        }
        return list.toArray(new String[list.size()]);
    }

    private HashMap<String, String> loadGroundTruth(){
        InputStream modelLabelsStream = null;
        try {
            modelLabelsStream = new FileInputStream(groundTruthPath);
        } catch (FileNotFoundException e) {
            e.printStackTrace();
            evaluationCallBacks.showErrorText("dataset label path not exit: " + groundTruthPath);
            return null;
        }
        HashMap<String, String> dataSetLabelsMap = new HashMap<>();
        BufferedReader inputStream = new BufferedReader(new InputStreamReader(modelLabelsStream));
        String line;
        Pattern p = Pattern.compile("(.+jpg|.+JPEG)\\s+(.+$)");
        while (true) {
            try {
                if ((line = inputStream.readLine()) == null)
                    break;
                Matcher m = p.matcher(line);
                if(m.find()) {
                    dataSetLabelsMap.put(m.group(1), m.group(2));
                }
            } catch (IOException e) {
                e.printStackTrace();
            }

        }
        return dataSetLabelsMap;
    }

    public int[] topKLabels(float[] output, int startPost, int length, int K) {
        int [] topk = new int[K];

        int endPost = startPost + length;
        Set<Integer> selected = new HashSet<>();
        for(int i=0; i<K; i++) {
            int maxIndex = startPost;
            float maxValue = output[startPost];
            for(int j=startPost; j < endPost; j++) {
                if(selected.contains(maxIndex) || (output[j] > maxValue && !selected.contains(j))) {
                    maxIndex = j;
                    maxValue = output[j];
                }
            }
            selected.add(maxIndex);
            topk[i] = maxIndex;
        }
        return topk;
    }

    @Override
    public boolean postProcessResult(ArrayList<File> inputImages) {
        resetResult();
        Log.d(TAG, "Path " + groundTruthPath + " " + modelLabelPath);
        String modelLabels[] = loadModelLabels();
        HashMap<String, String> groundTruthMap = loadGroundTruth();

        int imageNum = inputImages.size();
        Log.i(TAG, "postProcessResult doimage number: " + imageNum);
        String[] outputNames = PSNPEManager.getOutputTensorNames();
        for(int i = 0; i < imageNum; i++) {
            /* output:
             * <image1><image2>...<imageBulkSize>
             * split output and handle one by one.
             */
            float[] output = readOutput(i).get(outputNames[0]);

            int labelNum = modelLabels.length;

            String expectLabel = groundTruthMap.get(inputImages.get(i).getName());
            String actualLabel = modelLabels[topKLabels(output, 0, labelNum, 1)[0]];

            if(expectLabel.indexOf(actualLabel) != -1 || actualLabel.indexOf(expectLabel) != -1)
                top1Count++;

            int [] top5LabelIndex = topKLabels(output, 0, labelNum, 5);
            for(int k=0; k<top5LabelIndex.length; k++) {
                if(expectLabel.indexOf(modelLabels[top5LabelIndex[k]]) != -1
                || modelLabels[top5LabelIndex[k]].indexOf(expectLabel) != -1){
                    top5Count++;
                    break;
                }
            }

        }
        totNum = imageNumber;
        return true;
    }

    @Override
    public void resetResult() {
        totNum = 0;
        top1Count =0;
        top5Count = 0;
    }

    @Override
    public void setResult(Result result) {
        ClassificationResult cresult = (ClassificationResult)result;
        cresult.setTop1((float)top1Count/totNum);
        cresult.setTop5((float)top5Count/totNum);
        Log.d(TAG, "Top1Count: " + top1Count + " Top5Count: " +top5Count + " Total: " + totNum + "top5 " + cresult.getTop5());
    }

    @Override
    public void getOutputCallback(String fileName, Map<String, float[]> outputs) {
        String modelLabels[] = loadModelLabels();
        HashMap<String, String> groundTruthMap = loadGroundTruth();
        if(outputs.size() == 0) {
            Log.e(TAG, "output data is null");
            evaluationCallBacks.showErrorText("output result is empty");
            return;
        }

        for(Map.Entry<String, float[]> output : outputs.entrySet()) {
            String outputLayerName = output.getKey();
            float[] outputData = output.getValue();
            if (outputData == null) {
                Log.e(TAG, "output data is null");
                evaluationCallBacks.showErrorText("Layer: " + outputLayerName + " contains null data");
                return;
            }

            int labelNum = modelLabels.length;

            String expectLabel = groundTruthMap.get(fileName);
            String actualLabel = modelLabels[topKLabels(outputData, 0, labelNum, 1)[0]];

            if (expectLabel.indexOf(actualLabel) != -1 || actualLabel.indexOf(expectLabel) != -1)
                top1Count++;

            int[] top5LabelIndex = topKLabels(outputData, 0, labelNum, 5);
            for (int k = 0; k < top5LabelIndex.length; k++) {
                if (expectLabel.indexOf(modelLabels[top5LabelIndex[k]]) != -1
                        || modelLabels[top5LabelIndex[k]].indexOf(expectLabel) != -1) {
                    top5Count++;
                    break;
                }
            }
        }
        totNum ++;
        Log.i(TAG, "Async output postprocessor finished");
    }
}
