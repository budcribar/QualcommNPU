/*
 * Copyright (c) 2019-2023 Qualcomm Technologies, Inc.
 * All Rights Reserved.
 * Confidential and Proprietary - Qualcomm Technologies, Inc.
 */
package com.qualcomm.qti.psnpedemo.processor;

import com.qualcomm.qti.psnpedemo.networkEvaluation.Result;
import com.qualcomm.qti.psnpe.PSNPEManager;
import com.qualcomm.qti.psnpedemo.components.BenchmarkApplication;
import com.qualcomm.qti.psnpedemo.networkEvaluation.SegmentationResult;
import com.qualcomm.qti.psnpedemo.networkEvaluation.EvaluationCallBacks;
import com.qualcomm.qti.psnpedemo.networkEvaluation.ModelInfo;
import com.qualcomm.qti.psnpedemo.utils.MathUtils;
import com.qualcomm.qti.psnpedemo.utils.Util;

import java.io.File;
import java.util.ArrayList;
import java.util.Map;
import android.util.Log;
import java.util.List;


public class SegmentationPostProcessor extends PostProcessor {
    private static String TAG = "SegmentationPostProcessor";
    private static final String[] conf = {"TP","FP","FN"};
    private  double GlobalAcc = 0.0;
    private  double MeanIOU = 0.0;
    private  double MeanAccuracy = 0.0;
    private  double MeanPrecision = 0.0;
    private  double MeanF1Score = 0.0;
    private int LABEL_NUM = 21;
    private int BASIC_METRICS_NUM = 3;
    private int GRAY_NUM = 255;
    private int FIXED_HEIGHT = 0;
    private int FIXED_WIDTH = 0;
    private String packagePath;
    private String groundTruthPath;
    public static List<int[][]> confMatrixs = new ArrayList<>();

    public SegmentationPostProcessor(EvaluationCallBacks evaluationCallBacks, ModelInfo modelInfo, int imageNumber) {
        super(imageNumber);
        String modelName = modelInfo.getModelName();
        String dataSetName = modelInfo.getDataSetName();
        this.packagePath = BenchmarkApplication.getCustomApplicationContext().getExternalFilesDir("").getAbsolutePath();
        this.groundTruthPath = packagePath + "/datasets/" + dataSetName + "/SegmentationClass/";
    }

    @Override
    public boolean postProcessResult(ArrayList<File> inputImages) {
        int imageNum = inputImages.size();
        int[] inputDims = PSNPEManager.getInputDimensions();
        int imgDimension=inputDims[1];
        if(imgDimension == 512) {
            FIXED_HEIGHT = imgDimension;
            FIXED_WIDTH = imgDimension;
        } else if (imgDimension == 513) {
            FIXED_HEIGHT = imgDimension;
            FIXED_WIDTH = imgDimension;
        }

        int[] labels = new int[LABEL_NUM];
        for(int i = 0; i < LABEL_NUM; i++){
            labels[i] = i;
        }
        //List<int[][]> confMatrixs = new ArrayList<>();

        String[] outputNames = PSNPEManager.getOutputTensorNames();
        for(int imageIndex = 0; imageIndex < imageNum; imageIndex++) {
            float[] batchOutput = readOutput(imageIndex).get(outputNames[0]);
            String bulkImagePath = (inputImages.get(imageIndex)).getPath();
            File bulkImageFile =new File(bulkImagePath.trim());
            String bulkImageFilename = bulkImageFile.getName();
            String annoName=bulkImageFilename.substring(bulkImageFilename.lastIndexOf(".") );
            String annoImgFilename=bulkImageFilename.substring(0, bulkImageFilename.length()-annoName.length());
            String annoImgPath =groundTruthPath+annoImgFilename+".png";
            int[][] ConfusionMat = performIOU(annoImgPath, batchOutput);
            Log.i(TAG, "processing image on performIOU: " + annoImgPath);
            if (ConfusionMat != null) {
                confMatrixs.add(ConfusionMat);
            }
        }

        int[][] confMatrix = new int[GRAY_NUM][BASIC_METRICS_NUM];
        for(int i = 0; i < confMatrix.length; i++) {
            for(int j = 0; j < confMatrix[0].length; j++) {
                confMatrix[i][j] = 0;
            }
        }
        for(int i = 0; i < confMatrixs.size(); i++){
            int[][] matrix = confMatrixs.get(i);
            for(int k = 0; k < matrix.length; k++) {
                for(int j = 0; j < matrix[0].length; j++) {
                    confMatrix[k][j] += matrix[k][j];
                }
            }
        }
        double[] result = MathUtils.calSegIndex(confMatrix, labels);
        GlobalAcc = result[0];
        MeanIOU = result[1];
        MeanAccuracy = result[2];
        MeanPrecision = result[3];
        MeanF1Score = result[4];
        return true;
    }

    private int[][] performIOU(String imgPath, float[] resultVec) {
        int height, width;
        int[][] annoImage = Util.readImageToPmode(imgPath);
        height = annoImage.length;
        width = annoImage[0].length;
        int[][] result = Util.getResizedResultImage(resultVec,height, width,FIXED_HEIGHT, FIXED_WIDTH);
        if(annoImage == null || result == null) {
            return null;
        }

        int labelMin = Math.min(MathUtils.min(annoImage), MathUtils.min(result));
        labelMin = Math.min(labelMin, 0);

        int labelMax = Math.max(MathUtils.max(annoImage),MathUtils.max(result));
        labelMax = Math.max(labelMax, LABEL_NUM);

        int[] labels = new int[labelMax - labelMin];
        for(int i = 0; i < labels.length; i++) {
            labels[i] = labelMin++;
        }
        return MathUtils.getConfusionMat(annoImage, result, labels,conf );
    }

    @Override
    public void setResult(Result result) {
        SegmentationResult sgresult = (SegmentationResult) result;
        sgresult.setGlobalAcc(GlobalAcc);
        sgresult.setMeanIOU(MeanIOU);
        sgresult.setMeanAccuracy(MeanAccuracy);
        sgresult.setMeanPrecision(MeanPrecision);
        sgresult.setMeanF1Score(MeanF1Score);

    }

    @Override
    public void resetResult(){
        for(int i = 0; i < confMatrixs.size(); i++){
            int[][] matrix = confMatrixs.get(i);
            for(int k = 0; k < matrix.length; k++){
                for(int j = 0; j < matrix[0].length; j++){
                    matrix[k][j] = 0;
                }
            }
        }
    }

    @Override
    public void getOutputCallback(String fileName, Map<String, float[]> outputs) {

    }
}
