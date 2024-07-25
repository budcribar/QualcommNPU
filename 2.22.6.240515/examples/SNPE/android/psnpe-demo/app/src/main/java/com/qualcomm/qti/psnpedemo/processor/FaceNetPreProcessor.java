/*
 * Copyright (c) 2021-2022 Qualcomm Technologies, Inc.
 * All Rights Reserved.
 * Confidential and Proprietary - Qualcomm Technologies, Inc.
 */
package com.qualcomm.qti.psnpedemo.processor;

import android.util.Log;

import com.qualcomm.qti.psnpe.PSNPEManager;
import com.qualcomm.qti.psnpedemo.utils.Util;

import java.io.File;
import java.util.HashMap;

public class FaceNetPreProcessor extends PreProcessor{
    private static String TAG = FaceNetPreProcessor.class.getSimpleName();
    @Override
    public HashMap<String, float[]> preProcessData(File data) {
        String dataName = data.getName();
        if(dataName.toLowerCase().contains(".raw")){
            HashMap<String, float[]> outputMap = new HashMap<String, float[]>();
            String[] key = PSNPEManager.getInputTensorNames();
            outputMap.put(key[0],preProcessRaw(data));
            return outputMap;
        }
        else {
            Log.e(TAG, "data format invalid, dataName: " + dataName);
            return null;
        }
    }

    private float [] preProcessRaw(File data){
        int[] dimensions = PSNPEManager.getInputDimensions();
        int dataSize = 1 * dimensions[1] * dimensions[2] * dimensions[3];
        float[] floatArray = Util.readFloatArrayFromFile(data);
        if(floatArray.length != dataSize){
            Log.e(TAG, String.format("Wrong input data size: %d. Expect %d.", floatArray.length, dataSize));
            return null;
        }
        return floatArray;
    }
}
