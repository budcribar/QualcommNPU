/*
 * Copyright (c) 2022 Qualcomm Technologies, Inc.
 * All Rights Reserved.
 * Confidential and Proprietary - Qualcomm Technologies, Inc.
 */
package com.qualcomm.qti.psnpedemo.processor;
import android.util.Log;
import com.qualcomm.qti.psnpedemo.utils.Util;

import java.io.File;
import java.util.HashMap;

public class MobilebertPreProcessor extends PreProcessor {
    private static String TAG = MobilebertPreProcessor.class.getSimpleName();
    @Override
    public HashMap<String, float[]> preProcessData(File data) {
        File[] fileList = data.listFiles();
        assert fileList != null;
        File InputIdsRaw = null;
        File InputMaskRaw= null;
        File SegmentIdsRaw= null;
        for (File file : fileList) {
            String dataName = file.getName();
            if (dataName.toLowerCase().contains("input_ids.raw")) {
                InputIdsRaw = file;
            }
            if (dataName.toLowerCase().contains("input_mask.raw")) {
                InputMaskRaw = file;
            }
            if (dataName.toLowerCase().contains("segment_ids.raw")) {
                SegmentIdsRaw = file;
            }
        }
        HashMap<String, float[]> outputMap = new HashMap<String, float[]>();
        //bert model accept 3 inputs
        //bert/embeddings/ExpandDims:0:=squad11_75_question_float32/1000000000/input_ids.raw
        //input_mask:0:=squad11_75_question_float32/1000000000/input_mask.raw
        //segment_ids:0:=/squad11_75_question_float32/1000000000/segment_ids.raw
        String dataName = InputIdsRaw.getName();
        if(dataName.toLowerCase().contains(".raw")){
            outputMap.put("bert/embeddings/ExpandDims:0",preProcessRaw(InputIdsRaw));
            outputMap.put("input_mask:0",preProcessRaw(InputMaskRaw));
            outputMap.put("segment_ids:0",preProcessRaw(SegmentIdsRaw));
            return outputMap;
        }
        else {
            Log.e(TAG, "data format invalid, dataName: " + dataName);
            return null;
        }
    }
    private float [] preProcessRaw(File data){
        float[] floatArray = Util.readFloatArrayFromFile(data);
        return floatArray;
    }
}
