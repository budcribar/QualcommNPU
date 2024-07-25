/*
 * Copyright (c) 2019-2022 Qualcomm Technologies, Inc.
 * All Rights Reserved.
 * Confidential and Proprietary - Qualcomm Technologies, Inc.
 */
package com.qualcomm.qti.psnpedemo.processor;

import android.util.Log;

import com.qualcomm.qti.psnpedemo.utils.Util;
import com.qualcomm.qti.psnpe.PSNPEManager;

import java.io.File;
import java.util.HashMap;

public class MobileNetPreProcessor extends PreProcessor {
    private static String TAG = MobileNetPreProcessor.class.getSimpleName();
    @Override
    public HashMap<String, float[]> preProcessData(File data) {
        String dataName = data.getName().toLowerCase();
        if(!(dataName.contains(".jpg") || dataName.contains(".jpeg"))) {
            Log.d(TAG, "data format invalid, dataName: " + dataName);
            return null;
        }

        int [] tensorShapes = PSNPEManager.getInputDimensions(); // nhwc
        int length = tensorShapes.length;
        if(tensorShapes.length != 4 || tensorShapes[length-1] != 3) {
            Log.d(TAG, "data format should be BGR" + length + " " + tensorShapes[length-1]);
            return null;
        }

        double [] meanRGB = {128.0d, 128.0d, 128.0d};

        HashMap<String, float[]> outputMap = new HashMap<String, float[]>();
        String[] key = PSNPEManager.getInputTensorNames();
        outputMap.put(key[0],Util.imagePreprocess(data, tensorShapes[1], meanRGB, 128.0, false, 256));
        return outputMap;
    }
}
