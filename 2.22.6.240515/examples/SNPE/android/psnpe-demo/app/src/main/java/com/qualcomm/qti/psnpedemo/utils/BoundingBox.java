/*
 * Copyright (c) 2021 Qualcomm Technologies, Inc.
 * All Rights Reserved.
 * Confidential and Proprietary - Qualcomm Technologies, Inc.
 */
package com.qualcomm.qti.psnpedemo.utils;

import java.util.ArrayList;
import java.util.List;

public class BoundingBox {

    private float confidence;
    private String imageId;

    private ArrayList<Float> box;

    public BoundingBox(float confidence, String imageId, ArrayList<Float> box) {
        this.confidence = confidence;
        this.imageId = imageId;
        this.box = box;
    }

    public BoundingBox(float confidence, String imageId, ArrayList<Float> box, boolean used) {
        this.confidence = confidence;
        this.imageId = imageId;
        this.box = box;
    }

    public float getConfidence() {
        return confidence;
    }


    public String getImageId() {
        return imageId;
    }

    public ArrayList<Float> getBox() {
        return box;
    }


}
