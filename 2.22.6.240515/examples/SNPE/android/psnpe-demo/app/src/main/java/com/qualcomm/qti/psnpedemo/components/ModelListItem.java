/*
 * Copyright (c) 2019 Qualcomm Technologies, Inc.
 * All Rights Reserved.
 * Confidential and Proprietary - Qualcomm Technologies, Inc.
 */
package com.qualcomm.qti.psnpedemo.components;

public class ModelListItem {
    private String scenarioName;
    private String modelName;

    public ModelListItem(String scenarioName, String modelName) {
        this.scenarioName = scenarioName;
        this.modelName = modelName;
    }

    public String getScenarioName() {
        return scenarioName;
    }

    public String getModelName() {
        return modelName;
    }

}
