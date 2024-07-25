/*
 * Copyright (c) 2019-2023 Qualcomm Technologies, Inc.
 * All Rights Reserved.
 * Confidential and Proprietary - Qualcomm Technologies, Inc.
 */
package com.qualcomm.qti.psnpedemo.processor;

import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Color;
import android.util.Log;

import com.qualcomm.qti.psnpe.PSNPEManager;
import com.qualcomm.qti.psnpedemo.components.BenchmarkApplication;
import com.qualcomm.qti.psnpedemo.networkEvaluation.ModelInfo;
import com.qualcomm.qti.psnpedemo.utils.MathUtils;
import com.qualcomm.qti.psnpedemo.utils.Util;

import java.io.File;
import java.util.HashMap;


public class VDSRPreprocessor extends PreProcessor {
    private static final String TAG = VDSRPreprocessor.class.getSimpleName();
    private int radioSize;
    private final String groundTruthPath;

    public VDSRPreprocessor(ModelInfo modelInfo) {
        String truthRelPath = "datasets/"+modelInfo.getDataSetName()+"/GroundTruth";
        File truthDir = BenchmarkApplication.getCustomApplicationContext().getExternalFilesDir(truthRelPath);
        this.groundTruthPath = truthDir.getAbsolutePath();
        this.radioSize = 4;
    }

    @Override
    public HashMap<String, float[]> preProcessData(File data) {
        String dataName = data.getName().toLowerCase();
        if(!(dataName.contains(".jpg"))) {
            Log.e(TAG, "data format invalid, dataName: " + dataName);
            return null;
        }

        float result[] = getYLowDataRaw(data, radioSize);
        String inputPath = BenchmarkApplication.getCustomApplicationContext().getExternalFilesDir("input_list").getAbsolutePath();
        Util.write2file(inputPath + "/vdsr_input_list.txt", data.getName());

        HashMap<String, float[]> outputMap = new HashMap<String, float[]>();
        String[] key = PSNPEManager.getInputTensorNames();
        outputMap.put(key[0],result);
        return outputMap;
    }

    private float[] getYLowDataRaw(File imageName, int lowRatio){
        Bitmap originBitmap = null;
        try{
            originBitmap = BitmapFactory.decodeFile(imageName.getAbsolutePath());

        }catch (Exception e){
            Log.e(TAG, "Exception in image pre-processing: "+ e);
            return null;
        }

        int originImgWidth = originBitmap.getWidth();
        int originImgHeight = originBitmap.getHeight();
        int startX = Math.max((originImgWidth - 256) / 2, 0);
        int startY = Math.max((originImgHeight - 256) / 2, 0);
        int width = Math.min(originImgWidth, 256);
        int height = Math.min(originImgHeight, 256);
        Bitmap newBitmap = Bitmap.createBitmap(originBitmap, startX, startY, width, height, null, false);

        float R, G, B, Y;
        float[][] MatrixY = new float[height][width];
        for (int row = 0; row < height; row++) {
            for (int col = 0; col < width; col++) {
                int pixel = newBitmap.getPixel(col, row);
                R = Color.red(pixel);
                G = Color.green(pixel);
                B = Color.blue(pixel);
                /*convert RGB to YCbCr
                 * convert formula:
                 * Y = (0.256789 * R + 0.504129 * G + 0.097906 * B + 16.0)/255.0
                 * Cb = (-0.148223 * R - 0.290992 * G + 0.439215 * B + 128.0)/255.0
                 * Cr = (0.439215 * R  - 0.367789 * G - 0.071426 * B + 128.0)/255.0
                 * We only use Y channel here.
                 * */
                Y = (float)((0.256789 * R + 0.504129 * G + 0.097906 * B + 16.0)/255.0);
                MatrixY[row][col] = Y;
            }
        }

        if(width < 256 || height < 256){
            /*if input img size is smaller than 256*256, adjust it to 256*256*/
            MatrixY = MathUtils.matrixResizeLiner(MatrixY, 256, 256);
            width = 256;
            height = 256;
        }

        float[] labelMatrix = MathUtils.matrixReformat(MatrixY, MathUtils.Format.HW, MathUtils.Format.HW);
        String truthFileName = imageName.getName().replace("jpg", "raw");
        Util.writeArrayTofile(this.groundTruthPath + '/' + truthFileName, labelMatrix, true);

        int resize_width = width / lowRatio;
        int resize_height = height / lowRatio;

        float[][] resizeTmp = MathUtils.matrixResizeLiner(MatrixY, resize_height, resize_width);
        float[][] finalImg = MathUtils.matrixResizeLiner(resizeTmp, height, width);

        float[] inputData = MathUtils.matrixReformat(finalImg, MathUtils.Format.HW, MathUtils.Format.HW);

        return inputData;
    }
}
