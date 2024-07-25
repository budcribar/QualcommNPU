/*
 * Copyright (c) 2019-2021 Qualcomm Technologies, Inc.
 * All Rights Reserved.
 * Confidential and Proprietary - Qualcomm Technologies, Inc.
 */
package com.qualcomm.qti.psnpedemo.networkEvaluation;

public class  Result {
    private double fps;
    private double buildTime;
    private double inferenceTime;
    public void setBuildTime(double buildTime) {
        this.buildTime = buildTime;
    }
    public void setInferenceTime(double inferenceTime) {
        this.inferenceTime = inferenceTime;
    }
    public void setFPS(double fps){this.fps = fps;}
    public double getInferenceTime() {
        return inferenceTime;
    }
    public double getFPS(){
        return fps;
    }
    public void updateFromProfiler(TimeProfiler timeProfiler) {
        buildTime = timeProfiler.getTime(TimeProfiler.TIME_TYPE.BUILD_TIME) / 1000.0;
        inferenceTime = timeProfiler.getTime(TimeProfiler.TIME_TYPE.EXECUTE_TIME) / 1000.0;
    }
    public void clear(){
        fps = 0;
        buildTime = 0;
        inferenceTime = 0;
    }
    public Result() {
        fps = 0;
        buildTime = 0;
        inferenceTime = 0;
    }
    @Override
    public String toString() {
        String result = "";
        result = result + "FPS: " + getFPS()
                + "\nInference Time: "+ getInferenceTime() + "s\n";
        return result;
    }
}