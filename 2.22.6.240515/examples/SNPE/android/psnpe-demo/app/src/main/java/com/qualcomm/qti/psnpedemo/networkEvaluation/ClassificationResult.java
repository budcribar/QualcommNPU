/*
 * Copyright (c) 2021 Qualcomm Technologies, Inc.
 * All Rights Reserved.
 * Confidential and Proprietary - Qualcomm Technologies, Inc.
 */
package com.qualcomm.qti.psnpedemo.networkEvaluation;

public class ClassificationResult extends Result {

    private float top1;
    private float top5;
    private float tops;

    public void setTop1(float top1) {
        this.top1 = top1;
    }

    public void setTop5(float top5) {
        this.top5 = top5;
    }

    public void setTops(float tops) {
        this.tops = tops;
    }

    public float getTops() {
        return tops;
    }

    public float getTop1() {
        return top1;
    }

    public float getTop5() {
        return top5;
    }

    public ClassificationResult() {
        super();
        top1 = 0;
        top5 = 0;
        tops = 0;
    }

    @Override
    public void clear() {
        super.clear();
        top1 = 0;
        top5 = 0;
        tops = 0;
    }

    @Override
    public String toString() {
        String result = "";
        result = result + "FPS: " + super.getFPS()
                + "\nInference Time: " + super.getInferenceTime() + "s\nTop1: " + getTop1() * 100
                + "%\nTop5: " + getTop5() * 100 + "%\nTops: " + getTops();
        return result;
    }
}

