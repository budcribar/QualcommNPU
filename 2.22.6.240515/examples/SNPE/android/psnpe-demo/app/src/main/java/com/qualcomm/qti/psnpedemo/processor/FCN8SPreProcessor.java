/*
 * Copyright (c) 2019-2022 Qualcomm Technologies, Inc.
 * All Rights Reserved.
 * Confidential and Proprietary - Qualcomm Technologies, Inc.
 */
package com.qualcomm.qti.psnpedemo.processor;

import com.qualcomm.qti.psnpe.PSNPEManager;

import java.io.File;
import java.util.HashMap;

public class FCN8SPreProcessor extends PreProcessor {
    @Override
    public HashMap<String, float[]> preProcessData(File data) {
        HashMap<String, float[]> outputMap = new HashMap<String, float[]>();
        String[] key = PSNPEManager.getInputTensorNames();
        outputMap.put(key[0],new float[0]);
        return outputMap;
    }
}
