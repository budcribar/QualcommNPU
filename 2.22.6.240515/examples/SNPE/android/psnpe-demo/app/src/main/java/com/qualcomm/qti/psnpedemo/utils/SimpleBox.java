/*
 * Copyright (c) 2021 Qualcomm Technologies, Inc.
 * All Rights Reserved.
 * Confidential and Proprietary - Qualcomm Technologies, Inc.
 */
package com.qualcomm.qti.psnpedemo.utils;

import java.util.List;

public class SimpleBox {
    private List<Float> box;


    private boolean used;

    public SimpleBox(List<Float> box, boolean used) {
        this.box = box;
        this.used = used;
    }

    public List<Float> getBox() {
        return box;
    }

    public boolean isUsed() {
        return used;
    }

    public void setUsed(boolean used) {
        this.used = used;
    }

}
