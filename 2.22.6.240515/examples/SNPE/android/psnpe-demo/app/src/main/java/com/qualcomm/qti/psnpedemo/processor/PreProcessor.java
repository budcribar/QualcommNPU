/*
 * Copyright (c) 2019-2022 Qualcomm Technologies, Inc.
 * All Rights Reserved.
 * Confidential and Proprietary - Qualcomm Technologies, Inc.
 */
package com.qualcomm.qti.psnpedemo.processor;

import java.io.File;
import java.util.HashMap;

public abstract class PreProcessor {
    public abstract HashMap<String, float[]> preProcessData(File data);
}
