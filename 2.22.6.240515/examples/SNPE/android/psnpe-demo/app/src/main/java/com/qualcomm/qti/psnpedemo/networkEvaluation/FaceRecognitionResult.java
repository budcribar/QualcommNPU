/*
 * Copyright (c) 2021 Qualcomm Technologies, Inc.
 * All Rights Reserved.
 * Confidential and Proprietary - Qualcomm Technologies, Inc.
 */

package com.qualcomm.qti.psnpedemo.networkEvaluation;

public class FaceRecognitionResult extends Result {
    private double Accuracy;
    private double Threshold;

    public FaceRecognitionResult() {
        super();
        Accuracy = 0;
        Threshold = 0;
    }

    public double getAccuracy() {
        return Accuracy;
    }

    public void setAccuracy(double Accuracy) {
        this.Accuracy = Accuracy;
    }

    public double getThreshold() {
        return Threshold;
    }

    public void setThreshold(double Threshold) {
        this.Threshold = Threshold;
    }

    @Override
    public void clear() {
        super.clear();
        Accuracy = 0;
        Threshold = 0;
    }

    @Override
    public String toString() {
        String result = "";
        result = result + "FPS: " + super.getFPS()
                + "\nInference Time: " + super.getInferenceTime()
                + "\nBest Accuracy:" + getAccuracy()
                + "\nThreshold:" + getThreshold();
        return result;
    }
}
