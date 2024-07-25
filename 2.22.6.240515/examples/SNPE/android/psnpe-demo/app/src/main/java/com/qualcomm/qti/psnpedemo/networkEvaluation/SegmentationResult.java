/*
 * Copyright (c) 2021 Qualcomm Technologies, Inc.
 * All Rights Reserved.
 * Confidential and Proprietary - Qualcomm Technologies, Inc.
 */
package com.qualcomm.qti.psnpedemo.networkEvaluation;

public class SegmentationResult extends Result {
    private double GlobalAcc;
    private double MeanIOU;
    private double MeanAccuracy;
    private double MeanPrecision;
    private double MeanF1Score;

    public SegmentationResult() {
        super();
        GlobalAcc = 0;
        MeanIOU = 0;
        MeanAccuracy = 0;
        MeanPrecision = 0;
        MeanF1Score = 0;
    }

    public double getGlobalAcc() {
        return GlobalAcc;
    }

    public void setGlobalAcc(double GlobalAcc) {
        this.GlobalAcc = GlobalAcc;
    }

    public double getMeanIOU() {
        return MeanIOU;
    }

    public void setMeanIOU(double MeanIOU) {
        this.MeanIOU = MeanIOU;
    }

    public double getMeanAccuracy() {
        return MeanAccuracy;
    }

    public void setMeanAccuracy(double MeanAccuracy) {
        this.MeanAccuracy = MeanAccuracy;
    }

    public double getMeanPrecision() {
        return MeanPrecision;
    }

    public void setMeanPrecision(double MeanPrecision) {
        this.MeanPrecision = MeanPrecision;
    }

    public double getMeanF1Score() {
        return MeanF1Score;
    }

    public void setMeanF1Score(double MeanF1Score) {
        this.MeanF1Score = MeanF1Score;
    }

    @Override
    public void clear() {
        super.clear();
        GlobalAcc = 0;
        MeanIOU = 0;
        MeanAccuracy = 0;
        MeanPrecision = 0;
        MeanF1Score = 0;
    }

    @Override
    public String toString() {
        String result = "";
        result = result + "FPS: " + super.getFPS()
                + "\nInference Time: " + super.getInferenceTime()
                + "\nGlobalAcc:" + getGlobalAcc() + "\nMeanIOU:" + getMeanIOU()
                + "\nMeanAccuracy:" + getMeanAccuracy() + "\nMeanPrecision:" + getMeanPrecision()
                + "\nMeanF1Score:" + getMeanF1Score() ;
        return result;
    }
}
