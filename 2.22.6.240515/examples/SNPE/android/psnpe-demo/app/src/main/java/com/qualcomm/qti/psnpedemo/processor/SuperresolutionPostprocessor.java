/*
 * Copyright (c) 2019-2023 Qualcomm Technologies, Inc.
 * All Rights Reserved.
 * Confidential and Proprietary - Qualcomm Technologies, Inc.
 */
package com.qualcomm.qti.psnpedemo.processor;

import android.util.Log;

import com.qualcomm.qti.psnpe.PSNPEManager;
import com.qualcomm.qti.psnpedemo.components.BenchmarkApplication;
import com.qualcomm.qti.psnpedemo.networkEvaluation.Result;
import com.qualcomm.qti.psnpedemo.networkEvaluation.ModelInfo;
import com.qualcomm.qti.psnpedemo.networkEvaluation.SuperResolutionResult;
import com.qualcomm.qti.psnpedemo.utils.MathUtils;
import com.qualcomm.qti.psnpedemo.utils.Util;

import java.io.File;
import java.util.ArrayList;
import java.util.Map;

import static java.sql.Types.NULL;

public class SuperresolutionPostprocessor extends PostProcessor {
    private static final String TAG = SuperresolutionPostprocessor.class.getSimpleName();

    private double totalPSNR;
    private double totalSSIM;
    private double averagePSNR;
    private double averageSSIM;
    private final int imgHeight;
    private final int imgWidth;
    private final String dataSet;

    public SuperresolutionPostprocessor(ModelInfo modelInfo, int inputSize) {
        super(inputSize);
        totalPSNR = 0.0;
        totalSSIM = 0.0;
        averagePSNR = 0.0;
        averageSSIM = 0.0;
        imgHeight = 256;
        imgWidth = 256;
        dataSet = modelInfo.getDataSetName();
        String truthRelPath = "datasets/"+ dataSet +"/GroundTruth";
        File truthDir = BenchmarkApplication.getCustomApplicationContext().getExternalFilesDir(truthRelPath);
        this.groundTruthPath = truthDir.getAbsolutePath();
    }

    @Override
    public boolean postProcessResult(ArrayList<File> inputImages) {
        float [] groundTruth;
        String[] outputNames = PSNPEManager.getOutputTensorNames();
        for(int i=0; i<inputImages.size(); i++) {
            float[] output = readOutput(i).get(outputNames[0]);
            String truthFileName;
            String imgName = inputImages.get(i).getName();
            if (dataSet.equals("b100")) {
                truthFileName = imgName.replaceAll("\\.\\w+$", ".raw");
            }
            else if (dataSet.equals("Set5")) {
                truthFileName = imgName.replaceAll("\\.\\w+$", "_label.raw");
            }
            else {
                Log.e(TAG, String.format("Unsupported dataset \"%s\"", dataSet));
                return false;
            }
            groundTruth = Util.readFloatArrayFromFile(groundTruthPath + "/" + truthFileName);
            if(null == groundTruth){
                Log.e(TAG, "postProcessResult error: groundTruth is null");
                return false;
            }
            this.count.incrementAndGet();
            this.totalPSNR += computePSNR(groundTruth, output);
            this.totalSSIM += computeSSIM(groundTruth, output, 7);
        }
        this.averagePSNR = this.totalPSNR / this.count.doubleValue();
        this.averageSSIM = this.totalSSIM / this.count.doubleValue();
        return true;
    }

    @Override
    public void setResult(Result result) {
        SuperResolutionResult rresult = (SuperResolutionResult)result;
        rresult.setPSNR(averagePSNR);
        rresult.setSSIM(averageSSIM);
    }

    @Override
    public void resetResult(){}

    @Override
    public void getOutputCallback(String fileName, Map<String, float[]> outputs) {
        float [] output;
        float [] groundTruth;
        if(outputs.size() == 0){
            Log.e(TAG, "getOutputCallback error: outputMap is null");
            return;
        }
        output = outputs.get(PSNPEManager.getOutputTensorNames()[0]);
        if(null == output){
            Log.e(TAG, "getOutputCallback error: output is null");
            return;
        }

        String truthFileName = fileName.replace("jpg", "raw");
        groundTruth = Util.readFloatArrayFromFile(this.groundTruthPath + "/" + truthFileName);
        if(null == groundTruth){
            Log.e(TAG, "postProcessResult error: groundTruth is null");
            return;
        }

        this.totalPSNR = computePSNR(groundTruth, output);
        this.totalSSIM = computeSSIM(groundTruth, output, 7);
        this.averagePSNR = this.totalPSNR / this.count.doubleValue();
        this.averageSSIM = this.totalSSIM / this.count.doubleValue();
    }

    @Override
    public void clearOutput(){
        super.clearOutput();
        Util.delete(new File(groundTruthPath));
    }

    /*
     * Calculate MSE(Mean Square Error) between img1 and img2
     * */
    public double computeMSE(float[] img1, float[] img2) {
        if(img1.length != img2.length) {
            Log.e(TAG, "mse computing error with mismatch length of img1 and img2");
            return NULL;
        }
        double[] square = new double[img1.length];
        for (int i = 0; i < img1.length; i++) {
            square[i] = (img1[i] - img2[i]) * (img1[i] - img2[i]);
        }
        return MathUtils.mean(square);
    }

    /*
     * Calculate SSIM(Structural SIMilarity) between img1 and img2.
     * */
    public double computeSSIM(float[] img1, float[] img2, int windowSize) {
        int length = imgWidth * imgHeight;
        double[] img1_double = new double[img1.length];
        double[] img2_double = new double[img2.length];
        for (int i = 0; i < img1.length; ++i) {
            img1_double[i] = img1[i];
        }
        for (int i = 0; i < img2.length; ++i) {
            img2_double[i] = img2[i];
        }

        // means of img1
        double[] ux = uniformFilter1d(img1_double, windowSize);
        // means of img2
        double[] uy = uniformFilter1d(img2_double, windowSize);

        int ndim = 1;
        double NP = Math.pow(windowSize, ndim);
        double cov_norm = NP / (NP - 1);
        double[] uxx = uniformFilter1d(MathUtils.matrixMul(img1_double, img1_double), windowSize);
        double[] uyy = uniformFilter1d(MathUtils.matrixMul(img2_double, img2_double), windowSize);
        double[] uxy = uniformFilter1d(MathUtils.matrixMul(img1_double, img2_double), windowSize);
        double[] vx = new double[length];
        double[] vy = new double[length];
        double[] vxy = new double[length];
        for (int i = 0; i < length; i++) {
            // variances of img1
            vx[i] = cov_norm * (uxx[i] - ux[i] * ux[i]);
            // variances of img2
            vy[i] = cov_norm * (uyy[i] - uy[i] * uy[i]);
            // covariances of img1 and img2
            vxy[i] = cov_norm * (uxy[i] - ux[i] * uy[i]);
        }

        int data_range = 2;
        double K1 = 0.01;
        double K2 = 0.03;
        double C1 = Math.pow(K1 * data_range, 2);
        double C2 = Math.pow(K2 * data_range, 2);

        /* calculate all SSIM for img1 and img2 */
        double[] allSSIM = new double[length];
        for (int i = 0; i < length; i++) {
            double luminance = (2 * ux[i] * uy[i] + C1) / (ux[i] * ux[i] + uy[i] * uy[i] + C1);
            double contrast =  (2 * vxy[i] + C2) / (vx[i] + vy[i] + C2);
            allSSIM[i] = luminance * contrast;
        }

        int pad = (windowSize - 1) / 2;
        double[] croppedSSIM = new double[allSSIM.length - pad * 2];
        System.arraycopy(allSSIM, pad, croppedSSIM, 0, croppedSSIM.length);
        return MathUtils.mean(croppedSSIM);
    }

    /*
     * Calculate PSNR(Peak Signal to Noise Ratio) of img1 and img2.
     * */
    public double computePSNR(float[] im1, float[] im2) {
        double mse = computeMSE(im1, im2);
        return 10 * (Math.log10(1.0/mse));
    }

    /*
     * Calculate a 1-D minimum uniform filter
     * */
    private double[] uniformFilter1d(double[] input, int windowSize) {
        double[] output = new double[input.length];
        double[] paddingInput = new double[input.length + 2*(windowSize/2)];
        int start = 0;
        int end = 0;
        for (int i = 0; i < input.length + windowSize - 1; i++) {
            if(i < windowSize/2) {
                paddingInput[i] = input[windowSize/2 -1 - i];
            }
            else if(i >= input.length + windowSize/2) {
                paddingInput[i] = input[input.length -1 - end];
                end++;
            }
            else {
                paddingInput[i] = input[start++];
            }
        }

        start = windowSize /2;

        for (int i = 0; i < input.length; i++) {
            double average = 0.0;
            for (int j = 0; j <= windowSize/2; j++) {
                average += paddingInput[start - j];
            }
            for (int k = 1; k <= windowSize/2; k++) {
                average += paddingInput[start + k];
            }
            output[i] = average / windowSize;
            start++;
        }
        return output;
    }
}
