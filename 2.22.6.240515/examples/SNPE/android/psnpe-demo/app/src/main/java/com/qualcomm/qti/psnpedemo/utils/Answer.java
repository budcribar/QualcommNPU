/*
 * Copyright (c) 2022 Qualcomm Technologies, Inc.
 * All Rights Reserved.
 * Confidential and Proprietary - Qualcomm Technologies, Inc.
 */

package com.qualcomm.qti.psnpedemo.utils;

public class Answer {
    public Pos pos;
    public String text;

    public Answer(String text, Pos pos) {
        this.text = text;
        this.pos = pos;
    }

    public Answer(String text, int start, int end, float logit) {
        this(text, new Pos(start, end, logit));
    }

    public static class Pos implements Comparable<Pos> {
        public int start;
        public int end;
        public float logit;

        @Override
        public int compareTo(Pos other) {
            return Float.compare(other.logit, this.logit);
        }

        public Pos(int start, int end, float logit) {
            this.start = start;
            this.end = end;
            this.logit = logit;
        }
    }
}