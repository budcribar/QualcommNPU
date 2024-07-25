/*
 * Copyright (c) 2019-2020 Qualcomm Technologies, Inc.
 * All Rights Reserved.
 * Confidential and Proprietary - Qualcomm Technologies, Inc.
 */
package com.qualcomm.qti.psnpedemo.networkEvaluation;

import android.util.Log;

import java.time.Duration;
import java.time.Instant;

public class TimeProfiler {
    private static String TAG = TimeProfiler.class.getSimpleName();
    enum TIME_TYPE {
        BUILD_TIME,
        EXECUTE_TIME,
        TIME_TYPE_MAX
    }
    public TimeProfiler(boolean accumulate){
        this.accumulate = accumulate;
        int timeTypes = TIME_TYPE.TIME_TYPE_MAX.ordinal();
        times = new TimeUnit[timeTypes];
        for(int i=0; i < timeTypes; i++) {
            times[i] = new TimeUnit();
        }
    }

    static class TimeUnit {
        long start;
        long end;
        double time;
        TimeUnit() {
            time = 0;
        }
    }


    private boolean accumulate;
    private TimeUnit[] times;

    public void setAccumulate(boolean accumulate) {
        this.accumulate = accumulate;
    }

    public void startProfile(TIME_TYPE timeType) {
        if(times == null) {
            Log.e(TAG, "TimeProfile still not initialized;");
            return ;
        }

        if(timeType.ordinal() >= times.length) {
            Log.e(TAG, "Out of range: TIME_TYPE: " + timeType.ordinal() + "while time length" + times.length);
            return;
        }

        times[timeType.ordinal()].start = System.nanoTime();
    }

    public void endProfile(TIME_TYPE timeType) {
        long temp = System.nanoTime();
        if (times == null) {
            Log.e(TAG, "TimeProfile still not initialized;");
            return;
        }

        if(timeType.ordinal() >= times.length) {
            Log.e(TAG, "Out of range: TIME_TYPE: " + timeType.ordinal() + "while time length" + times.length);
            return;
        }

        times[timeType.ordinal()].end = temp;
        long increaseTime = times[timeType.ordinal()].end - times[timeType.ordinal()].start;
        if(accumulate) {
            times[timeType.ordinal()].time += increaseTime;
        }
        else
            times[timeType.ordinal()].time = increaseTime;
    }

    public double getTime(TIME_TYPE timeType) {
        if(timeType.ordinal() >= times.length) {
            Log.e(TAG, "Invalid time type");
            return 0;
        }
        return times[timeType.ordinal()].time / 1000000.0;
    }

    public void reset() {
        for(int i = 0; i < times.length; i++) {
            times[i].time = 0;
        }
    }
}
