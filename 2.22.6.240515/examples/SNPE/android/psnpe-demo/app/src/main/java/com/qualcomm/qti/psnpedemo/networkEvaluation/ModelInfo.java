/*
 * Copyright (c) 2019-2023 Qualcomm Technologies, Inc.
 * All Rights Reserved.
 * Confidential and Proprietary - Qualcomm Technologies, Inc.
 */
package com.qualcomm.qti.psnpedemo.networkEvaluation;

import android.util.Log;


import java.util.HashMap;
import java.util.Map;

public class ModelInfo {
    private static String TAG = ModelInfo.class.getSimpleName();
    private String scenarioName;
    private String modelName;
    private String dataSetName;
    private static HashMap<String, String[]> dataSetKey = new HashMap<String, String[]>(){
        {
            // init classification key
            String[] imagenetKey = {"inception", "resnet34", "resnet50", "mobilenet_v1", "mobilenet_v2", "vgg"};
            put("classificationData", imagenetKey);
            // init coco key
            String[] cocoKey = {"fcn8s","v1ssd","yolo_v3"};
            put("coco", cocoKey);
            // init voc key
            String[] vocKey = {"v2ssd","deeplabv3"};
            put("voc", vocKey);
            // init b100 key
            String[] b100Key = {"vdsr"};
            put("b100", b100Key);
            // init flw key
            String[] lfwKey = {"facenet"};
            put("lfw", lfwKey);
            // init squad key
            String[] squadkey = {"bert"};
            put("squad", squadkey);
            // init Set5 key
            String[] set5Key = {"rdn"};
            put("Set5", set5Key);
        }
    };
    public ModelInfo(){
        scenarioName = "classification";
        modelName = "inceptionv3";
        dataSetName = getDataSetName(modelName);
    }

    public ModelInfo(String scenarioName, String modelName) {
        this.scenarioName = scenarioName;
        this.modelName = modelName;
        this.dataSetName = getDataSetName(modelName);
    }

    public void setScenarioName(String scenarioName) {
        this.scenarioName = scenarioName;
    }

    public void setModelName(String modelName) {
        this.modelName = modelName;
        this.dataSetName = getDataSetName(modelName);
    }

    public String getScenarioName() {
        return scenarioName;
    }

    public String getModelName() {
        return modelName;
    }

    public String getDataSetName() {
        return dataSetName;
    }

    private String getDataSetName(String modelName) {
        String modelNameLowerCase = modelName.toLowerCase();
        for(Map.Entry<String, String[]> entry: dataSetKey.entrySet()) {
            String[] dataSetKeyList = entry.getValue();
            String dataSetName = entry.getKey();
            for(int i=0 ; i<dataSetKeyList.length; i++) {
                if(modelNameLowerCase.contains(dataSetKeyList[i]) || dataSetKeyList[i].contains(modelNameLowerCase)) {
                    Log.d(TAG, "data set: " + dataSetName);
                    return dataSetName;
                }
            }
        }
        return "classificationData";
    }
}
