/*
 * Copyright (c) 2021-2023 Qualcomm Technologies, Inc.
 * All Rights Reserved.
 * Confidential and Proprietary - Qualcomm Technologies, Inc.
 */
package com.qualcomm.qti.psnpedemo.networkEvaluation;

public class SuperResolutionResult extends Result {
    private double psnr;
    private double ssim;

    public SuperResolutionResult() {
        super();
        psnr = 0;
        ssim = 0;
    }

    public double getPSNR() {
        return psnr;
    }

    public void setPSNR(double PSNR) {
        this.psnr = PSNR;
    }

    public double getSSIM() {
        return ssim;
    }

    public void setSSIM(double SSIM) {
        this.ssim = SSIM;
    }

    @Override
    public void clear() {
        super.clear();
        psnr = 0;
        ssim = 0;
    }

    @Override
    public String toString() {
        String result = "";
        result = result + "FPS: " + super.getFPS()
                + "\nInference Time: " + super.getInferenceTime() + "s\n"
                + "\nPSNR: " + getPSNR()
                + "\nSSIM: " + getSSIM();
        return result;
    }
}
