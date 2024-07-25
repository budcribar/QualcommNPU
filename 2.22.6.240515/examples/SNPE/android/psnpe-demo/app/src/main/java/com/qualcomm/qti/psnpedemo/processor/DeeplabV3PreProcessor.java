/*
 * Copyright (c) 2019-2022 Qualcomm Technologies, Inc.
 * All Rights Reserved.
 * Confidential and Proprietary - Qualcomm Technologies, Inc.
 */
package com.qualcomm.qti.psnpedemo.processor;

import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Color;
import android.util.Log;

import com.qualcomm.qti.psnpedemo.components.BenchmarkApplication;
import com.qualcomm.qti.psnpedemo.utils.Util;
import com.qualcomm.qti.psnpe.PSNPEManager;

import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.util.HashMap;


public class DeeplabV3PreProcessor extends PreProcessor {
    private static String TAG = DeeplabV3PreProcessor.class.getSimpleName();

    @Override
    public HashMap<String, float[]> preProcessData(File data) {
        String dataName = data.getName().toLowerCase();
        if(!(dataName.contains(".jpg") || dataName.contains(".jpeg") || dataName.contains(".png"))) {
            Log.d(TAG, "data format invalid, dataName: " + dataName);
            return null;
        }

        int [] tensorShapes = PSNPEManager.getInputDimensions(); // nhwc
        int length = tensorShapes.length;
        if(tensorShapes.length != 4 || tensorShapes[length-1] != 3) {
            Log.d(TAG, "data format should be BGR");
            return null;
        }

        int inputSize = tensorShapes[1];
        int resize_target = 512;
        if(inputSize != resize_target) {
            Log.d(TAG, "inputSize should be 512, actual is" + inputSize);
            return null;
        }

        float[] pixelFloats = new float[inputSize * inputSize * 3];
        Bitmap img;
        try{
            img = BitmapFactory.decodeStream(new FileInputStream(data.getAbsolutePath()));
            //resize image
            int width = img.getWidth();
            int height = img.getHeight();
            int max_dim = Math.max(width, height);
            //Log.d(TAG, "Width " + width + " height" + height);
            if(max_dim > resize_target) {
                Log.d(TAG, "images size should not larger than " + resize_target + "w: " + width + "h: " + height);
                return null;
            }

            int z = 0;
            for (int y = 0; y < inputSize; y++) {
                for (int x = 0; x < inputSize; x++) {
                    int rgb;

                    if(x >= width || y >= height) {
                        rgb = Color.rgb(128, 128, 128);
                    }
                    else {
                        rgb = img.getPixel(x, y);
                    }

                    double b = (Color.blue(rgb) - 127.5) / 127.5;
                    double g = (Color.green(rgb) - 127.5) / 127.5;
                    double r = (Color.red(rgb) - 127.5) / 127.5;

                    pixelFloats[z++] = (float)r;
                    pixelFloats[z++] = (float)g;
                    pixelFloats[z++] = (float)b;

                }
            }
            img.recycle();
        } catch (IOException e) {
            e.printStackTrace();
        }
        String inputPath = BenchmarkApplication.getCustomApplicationContext().getExternalFilesDir("input_list").getAbsolutePath();
        Util.write2file(inputPath + "/deeplabv3_input_list.txt", data.getName());

        HashMap<String, float[]> outputMap = new HashMap<String, float[]>();
        String[] key = PSNPEManager.getInputTensorNames();
        outputMap.put(key[0],pixelFloats);
        return outputMap;
    }


}
