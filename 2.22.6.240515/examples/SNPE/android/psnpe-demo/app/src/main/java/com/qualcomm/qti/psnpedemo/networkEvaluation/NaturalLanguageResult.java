/*
 * Copyright (c) 2022 Qualcomm Technologies, Inc.
 * All Rights Reserved.
 * Confidential and Proprietary - Qualcomm Technologies, Inc.
 */
package com.qualcomm.qti.psnpedemo.networkEvaluation;

public class NaturalLanguageResult extends Result {

    private float F1;
    private float ExactMatch;

    public void setF1(float F1) {
        this.F1 = F1;
    }

    public void setExactMatch(float ExactMatch) {
        this.ExactMatch = ExactMatch;
    }

    public float getF1() {
        return F1;
    }

    public float getExactMatch() {
        return ExactMatch;
    }

    public NaturalLanguageResult() {
        super();
        F1 = 0;
        ExactMatch = 0;
    }

    @Override
    public void clear() {
        super.clear();
        F1 = 0;
        ExactMatch = 0;
    }

    @Override
    public String toString() {
        String result = "";
        result = result + "FPS: " + super.getFPS()
                + "\nInference Time: " + super.getInferenceTime() + "s\nF1: " + getF1() * 100
                + "\nExactMatch: " + getExactMatch() * 100 ;
        return result;
    }
}

