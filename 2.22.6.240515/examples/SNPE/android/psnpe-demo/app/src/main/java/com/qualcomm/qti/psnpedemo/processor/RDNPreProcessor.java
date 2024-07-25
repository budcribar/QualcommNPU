/*
 * Copyright (c) 2023 Qualcomm Technologies, Inc.
 * All Rights Reserved.
 * Confidential and Proprietary - Qualcomm Technologies, Inc.
 */
package com.qualcomm.qti.psnpedemo.processor;

import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Color;
import android.util.Log;
import java.io.File;
import java.io.FileInputStream;
import java.util.HashMap;

import com.qualcomm.qti.psnpe.PSNPEManager;
import com.qualcomm.qti.psnpedemo.components.BenchmarkApplication;
import com.qualcomm.qti.psnpedemo.networkEvaluation.ModelInfo;
import com.qualcomm.qti.psnpedemo.utils.MathUtils;
import com.qualcomm.qti.psnpedemo.utils.Util;


public class RDNPreProcessor extends PreProcessor{
    private static final String TAG = RDNPreProcessor.class.getSimpleName();
    private final String groundTruthPath;

    public RDNPreProcessor(ModelInfo modelInfo) {
        String truthRelPath = "datasets/"+ modelInfo.getDataSetName() +"/GroundTruth";
        File truthDir = BenchmarkApplication.getCustomApplicationContext().getExternalFilesDir(truthRelPath);
        groundTruthPath = truthDir.getAbsolutePath();
    }
    @Override
    public HashMap<String,float[]> preProcessData(File data) {
        String dataName = data.getName();
        float[] rawData;
        if (dataName.toLowerCase().contains(".png")) {
            rawData = preProcessImg(data);
        }
        else if(dataName.toLowerCase().contains(".raw")){
            rawData = preProcessRaw(data);
        }
        else {
            Log.e(TAG, "data format invalid, dataName: " + dataName);
            return null;
        }
        HashMap<String, float[]> inputData = new HashMap<>();
        String[] key = PSNPEManager.getInputTensorNames();
        inputData.put(key[0], rawData);
        return inputData;
    }

    private float [] preProcessRaw(File data){
        int[] dimensions = PSNPEManager.getInputDimensions();
        int dataSize = dimensions[1] * dimensions[2] * dimensions[3];
        float[] floatArray = Util.readFloatArrayFromFile(data);
        if(floatArray.length != dataSize){
            Log.e(TAG, String.format("Wrong input data size: %d. Expect %d.", floatArray.length, dataSize));
            return null;
        }
        return floatArray;
    }

    private float[] preProcessImg(File dataFile){
        Bitmap bitmap = BitmapFactory.decodeFile(dataFile.getAbsolutePath());
        int originImgWidth = bitmap.getWidth();
        int originImgHeight = bitmap.getHeight();
        float[][][] originMat = new float[3][originImgHeight][originImgWidth]; //format: CHW
        for (int y = 0; y < originImgHeight; ++y) {
            for (int x = 0; x < originImgWidth; ++x) {
                int pixel = bitmap.getPixel(x, y);
                // BGR
                originMat[2][y][x] = (float)Color.blue(pixel);
                originMat[1][y][x] = (float)Color.green(pixel);
                originMat[0][y][x] = (float)Color.red(pixel);
            }
        }

        float[][][] labelMat = generateLabel(originMat); //format: CHW

        int[] inputDims = PSNPEManager.getInputDimensions(); //inputDims:[B,H,W,C]
        int inputHeight = inputDims[1];
        int inputWidth = inputDims[2];
        int channel = inputDims[3];
        float[][][] inputMat; //format: CHW
        if (originImgHeight != inputHeight || originImgWidth != inputWidth) {
            inputMat = new float[channel][][];
            for (int c = 0; c < channel; ++c) {
                inputMat[c] = MathUtils.matrixResizeCUBIC(labelMat[c], inputHeight, inputWidth);
            }
            inputMat = MathUtils.round(inputMat);
        }
        else {
            inputMat = originMat;
        }

        inputMat = MathUtils.matrixDiv(inputMat, 255.0);
        //format: HWC
        float[] inputArray = MathUtils.matrixReformat(inputMat, MathUtils.Format.CHW, MathUtils.Format.HWC);

        labelMat = MathUtils.matrixDiv(labelMat, 255.0);
        saveLabel(labelMat, dataFile.getName());
        return inputArray;
    }

    // input format: CHW, output format: CHW
    private float[][][] generateLabel(float[][][] srcMat) {
        float[][][] labelMat;
        int channel = srcMat.length;
        int srcHeight = srcMat[0].length;
        int srcWidth = srcMat[0][0].length;
        String[] outputNames = PSNPEManager.getOutputTensorNames();
        int[] outputDims = PSNPEManager.getOutputDimensions(outputNames[0]); //format: BHWC
        int labelHeight = outputDims[1];
        int labelWidth = outputDims[2];
        if (srcHeight != labelHeight || srcWidth != labelWidth) {
            labelMat = new float[channel][][];
            for (int c = 0; c < channel; ++c) {
                labelMat[c] =  MathUtils.matrixResizeLiner(srcMat[c], labelHeight, labelWidth);
            }
            labelMat = MathUtils.round(labelMat);
        }
        else {
            labelMat = srcMat;
        }
        return labelMat;
    }

    // labelMat format: CHW
    private void saveLabel(float[][][] labelMat, String imgName) {
        String labelName = imgName.replaceAll("\\.\\w+$", "_label.raw");
        float[] data = MathUtils.matrixReformat(labelMat, MathUtils.Format.CHW, MathUtils.Format.HWC);
        Util.writeArrayTofile(this.groundTruthPath + '/' + labelName, data, true);
    }
}
