/*
 * Copyright (c) 2021 Qualcomm Technologies, Inc.
 * All Rights Reserved.
 * Confidential and Proprietary - Qualcomm Technologies, Inc.
 */
package com.qualcomm.qti.psnpedemo.networkEvaluation;

public class DetectionResult extends Result {

    private double map;

    public void setMap(double map) {
        this.map = map;
    }

    public double getMap() {
        return map;
    }

    public DetectionResult() {
        super();
        map = 0f;

    }

    @Override
    public void clear() {
        super.clear();
        map = 0f;
    }

    @Override
    public String toString() {
        String result = "";
        result = result + "FPS: " + super.getFPS()
                + "\nInference Time: " + super.getInferenceTime() + "s\nmAP: " + getMap();
        return result;
    }
}
