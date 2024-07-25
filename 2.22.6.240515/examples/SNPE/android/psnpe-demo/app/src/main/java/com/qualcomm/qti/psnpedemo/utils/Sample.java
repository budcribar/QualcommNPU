/*
 * Copyright (c) 2022 Qualcomm Technologies, Inc.
 * All Rights Reserved.
 * Confidential and Proprietary - Qualcomm Technologies, Inc.
 */
package com.qualcomm.qti.psnpedemo.utils;

import java.util.List;
import java.util.Map;

/** Feature to be fed into the Bert model. */
public final class Sample {
    public final int[] inputIds;
    public final int[] inputMask;
    public final int[] segmentIds;
    public final List<String> origTokens;
    public final List<String> tokTokens;
    public final Map<Integer, Integer> tokenToOrigMap;

    public Sample(
            List<Integer> inputIds,
            List<Integer> inputMask,
            List<Integer> segmentIds,
            List<String> origTokens,
            List<String> tokTokens,
            Map<Integer, Integer> tokenToOrigMap) {
        this.inputIds = inputIds.stream().mapToInt(Integer::intValue).toArray();
        this.inputMask = inputMask.stream().mapToInt(Integer::intValue).toArray();
        this.segmentIds = segmentIds.stream().mapToInt(Integer::intValue).toArray();
        this.origTokens = origTokens;
        this.tokTokens = tokTokens;
        this.tokenToOrigMap = tokenToOrigMap;
    }
}
