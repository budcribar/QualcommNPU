/*
 * Copyright (c) 2021-2023 Qualcomm Technologies, Inc.
 * All Rights Reserved.
 * Confidential and Proprietary - Qualcomm Technologies, Inc.
 */
package com.qualcomm.qti.psnpedemo.utils;

import android.util.Log;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

public class MathUtils {
    public final static String TAG = MathUtils.class.getSimpleName();
    public enum Format{
        HW,
        WH,
        HWC,
        CHW
    };

    public static double mean(double[] data) {
        if(data.length == 0) {
            return Double.NaN;
        }
        double mean = 0.0;
        for (double datum : data) {
            mean += datum;
        }
        mean = mean / (double)data.length;
        return mean;
    }

    public static double min(ArrayList<Double> nums){
        double minNum = Double.MAX_VALUE;
        for (double num : nums){
            if (num <= minNum){
                minNum = num;
            }
        }
        return minNum;
    }
    public static int min(int[][] num) {
        int min = Integer.MAX_VALUE;
        for(int i = 0; i < num.length; i++) {
            for(int j = 0; j < num[0].length; j++) {
                if(num[i][j] < min) {
                    min = num[i][j];
                }
            }
        }
        return min;
    }

    public static double max(ArrayList<Double> nums){
        double maxNum = 0;
        for (double num : nums){
            if (num >= maxNum){
                maxNum = num;
            }
        }
        return maxNum;
    }
    public static int max(int[][] num) {
        int max = Integer.MIN_VALUE;
        for(int i = 0; i < num.length; i++) {
            for(int j = 0; j < num[0].length; j++) {
                if(num[i][j] > max) {
                    max = num[i][j];
                }
            }
        }
        return max;
    }

    public static float[][][] round(float[][][] in) {
        float[][][] out = new float[in.length][in[0].length][in[0][0].length];
        for (int i = 0; i < in.length; ++i) {
            for (int j = 0; j < in[i].length; ++j) {
                for (int k = 0; k < in[i][j].length; ++k){
                    out[i][j][k] = (float) Math.round(in[i][j][k]);
                }
            }
        }
        return out;
    }

    public static double getAverage(List<Double> num) {
        if(num.size() == 0) {
            throw new NullPointerException();
        }
        double average = 0.0;
        for(int i = 0; i < num.size(); i++) {
            average += num.get(i);
        }
        return (double) average/num.size();
    }

    public static int[][] matrixReshape(float[] src ,int height, int width) {
        if(src.length == 0 || src.length != height * width) {
            Log.e(TAG, "matrix length=" + src.length + ", height=" + height + ", width=" + width);
            return null;
        }
        int [][] reshapedMatrix = new int[height][width];

        for(int i = 0; i < height; i++) {
            for(int j = 0; j < width; j++) {
                reshapedMatrix[i][j] = (int)src[i*width + j];
            }
        }
        return reshapedMatrix;
    }
    public static int[][] matrixReshapeAndCut(float[] matrix, int cut_height, int cut_width, int reshape_height, int reshape_width) {
        if(matrix.length == 0 || matrix.length != reshape_height * reshape_width) {
            Log.e(TAG, "matrix length=" + matrix.length + ", reshape height=" + reshape_height + ", reshape width=" + reshape_width);
            return null;
        }
        int [][] cutMatrix = new int[cut_height][cut_width];

        for(int i = 0; i < reshape_height; i++) {
            for(int j = 0; j < reshape_width; j++) {
                if(i >= cut_height || j >= cut_width)
                    continue;
                cutMatrix[i][j] = (int)matrix[i*reshape_width + j];
            }
        }
        return cutMatrix;
    }

    public static  int[][] getConfusionMat(int[][] anno, int[][] pred, int[] labels,String[] conf) {
        int[][] cf = new int[labels.length][3];
        for(int i = 0; i < labels.length; i++) {
            for(int j = 0; j < conf.length; j++) {
                cf[i][j] = 0;
            }
        }
        Set<String> resPos = new HashSet<>();
        Set<String> annoPos = new HashSet<>();
        Set<String> TP = new HashSet<>();
        Set<String> FN = new HashSet<>();
        Set<String> FP = new HashSet<>();

        for(int label: labels) {
            resPos.clear();
            annoPos.clear();
            for(int i = 0; i < pred.length; i++) {
                for(int j = 0; j < pred[0].length; j++) {
                    if(pred[i][j] == label) {
                        resPos.add("("+ i +","+ j +")");
                    }
                }
            }
            for(int i = 0; i < anno.length; i++) {
                for(int j = 0; j < anno[0].length; j++) {
                    if(anno[i][j] == label) {
                        annoPos.add("("+ i +","+ j +")");
                    }
                }
            }
            TP.clear();
            TP.addAll(resPos);
            TP.retainAll(annoPos);

            FN.clear();
            FN.addAll(annoPos);
            FN.removeAll(TP);

            FP.clear();
            FP.addAll(resPos);
            FP.removeAll(TP);

            cf[label][0] = TP.size();
            cf[label][1] = FP.size();
            cf[label][2] = FN.size();
        }
        return cf;
    }
    public static double[] calAverages(double[]precision, double[]recall, int[] labels) {
        double[] result = new double[2];
        List<Integer> precT = new ArrayList<Integer>();
        List<Integer> recallT = new ArrayList<Integer>();
        List<Double> precFinal = new ArrayList<Double>();
        List<Double> recallFinal = new ArrayList<Double>();
        for(int i = 0; i < labels.length; i++) {
            if(Double.isNaN(precision[i])) {
                precT.add(i);
            }
            if(Double.isNaN(recall[i])) {
                recallT.add(i);
            }
        }
        Set tagSet = new HashSet();
        for(int i = 0; i < labels.length; i++) {
            tagSet.add(i);
        }
        Set precTSet = new HashSet(precT);
        Set recallTSet = new HashSet(recallT);
        precTSet.addAll(recallTSet);
        tagSet.removeAll(precTSet);
        List<Object> finalList = Arrays.asList(tagSet.toArray());
        List<Integer> notCal = new ArrayList<Integer>();
        for(Object i:precTSet.toArray()) {
            notCal.add(labels[(Integer) i]);
        }
        for(int i = 0; i < finalList.size(); i++) {
            precFinal.add(precision[(Integer) finalList.get(i)]);
            recallFinal.add(recall[(Integer) finalList.get(i)]);
        }
        result[0] = getAverage(precFinal);
        result[1] = getAverage(recallFinal);
        return result;

    }
    public static  double[]  calSegIndex(int[][] confMatrix, int[] labels) {
        double[] GlobalACC = new double[confMatrix.length];
        double[] IOU = new double[confMatrix.length];
        double[] Recall = new double[confMatrix.length];
        double[] Prec = new double[confMatrix.length];
        double[] F1Score = new double[confMatrix.length];
        int[] TP = new int[confMatrix.length];
        int[] FP = new int[confMatrix.length];
        int[] FN = new int[confMatrix.length];
        for(int i = 0; i < TP.length; i++) {
            TP[i] = confMatrix[i][0];
        }
        for(int i = 0; i < FP.length; i++) {
            FP[i] = confMatrix[i][1];
        }
        for(int i = 0; i < FN.length; i++) {
            FN[i] = confMatrix[i][2];
        }
        for(int i = 0; i < confMatrix.length; i++) {
            if(TP[i]+FN[i] == 0) {
                GlobalACC[i] = Double.NaN;
            }
            else {
                GlobalACC[i] = (double)TP[i]/(TP[i]+FN[i]);
            }
            if(FP[i]+FN[i]+TP[i] == 0){
                IOU[i] = Double.NaN;
            }
            else{
                IOU[i] = (double)TP[i]/(FP[i]+FN[i]+TP[i]);
            }
            if(TP[i]+FN[i] == 0) {
                Recall[i] = Double.NaN;
            }
            else{
                Recall[i] = (double)TP[i]/(TP[i]+FN[i]);
            }
            if(TP[i]+FP[i] == 0){
                Prec[i] = Double.NaN;
            }
            else{
                Prec[i] = (double)TP[i]/(TP[i]+FP[i]);
            }
            if(Recall[i]+Prec[i] == 0) {
                F1Score[i] = Double.NaN;
            }
            else{
                F1Score[i] = (2*Prec[i]*Recall[i])/(Recall[i]+Prec[i]);
            }

        }
        double[] result;
        result= MathUtils.calAverages(Prec, Recall, labels);
        double averagePrec = result[0];
        result = MathUtils.calAverages(F1Score, Recall, labels);
        double averageF1Score = result[0];
        result = MathUtils.calAverages(IOU, Recall, labels);
        double averageIOU = result[0];
        result = MathUtils.calAverages(GlobalACC, Recall, labels);
        double averageGlobalAcc = result[0];
        double averageRecall = result[1];
        result= new double[]{averageGlobalAcc, averageIOU,averageRecall,averagePrec,averageF1Score};
        return result;
    }

    public static  float[][][] matrixDiv(float[][][] divided, double divisor) {
        int[] dims = new int[]{divided.length, divided[0].length, divided[0][0].length};
        float[][][] result = new float[dims[0]][dims[1]][dims[2]];
        for (int idx0 = 0; idx0 < dims[0]; ++idx0) {
            for (int idx1 = 0; idx1 < dims[1]; ++idx1) {
                for (int idx2 = 0; idx2 < dims[2]; ++idx2) {
                    result[idx0][idx1][idx2] = (float)(divided[idx0][idx1][idx2] / divisor);
                }
            }
        }
        return result;
    }

    public static double[] matrixMul(double[] input1, double[] input2){
        double[] output = new double[input1.length];
        for (int i = 0; i < output.length; ++i){
            output[i] = input1[i] * input2[i];
        }
        return output;
    }

    public static float[] matrixReformat(float[][][] src, Format srcFormat, Format dstFormat) {
        float[] dst;
        if (Format.CHW == srcFormat && Format.HWC == dstFormat) {
            int channel = src.length;
            int height = src[0].length;
            int width = src[0][0].length;
            dst = new float[height * width * channel];
            for (int h = 0; h < height; ++h) {
                for (int w = 0; w < width; ++w) {
                    for (int c = 0; c < channel; ++c) {
                        dst[h*width*channel + w*channel + c] = src[c][h][w];
                    }
                }
            }
        }
        else {
            throw new UnsupportedOperationException(String.format("Unsupported Reformat: %s -> %s", srcFormat, dstFormat));
        }
        return dst;
    }
    public static float[] matrixReformat(float[][] src, Format srcFormat, Format dstFormat) {
        float[] dst;
        if (Format.HW == srcFormat && Format.HW == dstFormat) {
            int height = src.length;
            int width = src[0].length;
            dst = new float[height * width];
            for (int h = 0; h < height; ++h) {
                for (int w = 0; w < width; ++w) {
                    dst[h*width + w] = src[h][w];
                }
            }
        }
        else if (Format.WH == srcFormat && Format.HW == dstFormat) {
            int width = src.length;
            int height = src[0].length;
            dst = new float[height * width];
            for (int h = 0; h < height; ++h) {
                for (int w = 0; w < width; ++w) {
                    dst[h*width + w] = src[w][h];
                }
            }
        }
        else {
            throw new UnsupportedOperationException(String.format("Unsupported Reformat: %s -> %s", srcFormat, dstFormat));
        }
        return dst;
    }

    public static float[][] matrixResizeLiner(float [][] src, int dstHeight, int dstWidth){
        if(null == src){
            throw new IllegalArgumentException("src buffer is null");
        }
        if(0 == src.length || 0 == src[0].length){
            throw new IllegalArgumentException(String.format("Wrong resize src buffer size(%d, %d)", dstWidth, dstHeight));
        }
        if(0 == dstWidth || 0 == dstHeight){
            throw new IllegalArgumentException(String.format("Wrong resize dstSize(%d, %d)", dstWidth, dstHeight));
        }

        int srcHeight = src.length;
        int srcWidth = src[0].length;
        float[][] dst = new float[dstHeight][dstWidth];

        double scaleX = (double)srcWidth / (double)dstWidth;
        double scaleY = (double)srcHeight / (double)dstHeight;

        for(int dstY = 0; dstY < dstHeight; ++dstY){
            double fy = ((double)dstY + 0.5) * scaleY - 0.5;
            int sy = (int)fy;
            fy -= sy;
            if(sy < 0){
                fy = 0.0; sy = 0;
            }
            if(sy >= srcHeight - 1){
                fy = 0.0; sy = srcHeight - 2;
            }

            for(int dstX = 0; dstX < dstWidth; ++dstX){
                double fx = ((double)dstX + 0.5) * scaleX - 0.5;
                int sx = (int)fx;
                fx -= sx;
                if(sx < 0){
                    fx = 0.0; sx = 0;
                }
                if(sx >= srcWidth - 1){
                    fx = 0.0; sx = srcWidth - 2;
                }

                dst[dstY][dstX] = (float) ((1.0-fx) * (1.0-fy) * src[sy][sx]
                        + fx * (1.0-fy) * src[sy][sx+1]
                        + (1.0-fx) * fy * src[sy+1][sx]
                        + fx * fy * src[sy+1][sx+1]);
            }
        }

        return dst;
    }

    public static float[][] matrixResizeCUBIC(float [][] src, int dstHeight, int dstWidth){
        int srcHeight = src.length;
        int srcWidth = src[0].length;
        float[][] dst = new float[dstHeight][dstWidth];
        double scaleX = (double)srcWidth / (double)dstWidth;
        double scaleY = (double)srcHeight / (double)dstHeight;
        int ksize = 4;

        int[][] xofs = new int[dstWidth][ksize];
        int[][] yofs = new int[dstHeight][ksize];
        double[][] alpha = new double[dstWidth][ksize];
        double[][] beta = new double[dstHeight][ksize];

        for (int dx = 0; dx < dstWidth; ++dx) {
            double fx = ((double)dx + 0.5) * scaleX - 0.5;
            int sx = (int)fx;
            fx -= sx;
            double[] cbuf = interpolateCubic(fx);
            for(int k = 0; k < ksize; ++k) {
                xofs[dx][k] = clip(sx - 1 + k, 0, srcWidth - 1);
                alpha[dx][k] = cbuf[k];
            }
        }
        for (int dy = 0; dy < dstHeight; ++dy) {
            double fy = ((double)dy + 0.5) * scaleY - 0.5;
            int sy = (int)fy;
            fy -= sy;
            double[] cbuf = interpolateCubic(fy);
            for(int k = 0; k < ksize; ++k) {
                yofs[dy][k] = clip(sy - 1 + k, 0, srcHeight - 1);
                beta[dy][k] = cbuf[k];
            }
        }

        for (int dy = 0; dy < dstHeight; ++dy) {
            for (int dx = 0; dx < dstWidth; ++dx) {
                for (int i = 0; i < ksize; ++i) {
                    for (int j = 0; j < ksize; ++j) {
                        int xi = xofs[dx][i], yj = yofs[dy][j];
                        dst[dy][dx] += alpha[dx][i] * beta[dy][j] * src[yj][xi];
                    }
                }
            }
        }

        return dst;
    }
    private static double[] interpolateCubic(double x) {
        double[] coeffs = new double[4];
        double A = -0.75;
        coeffs[0] = ((A*(x + 1) - 5*A)*(x + 1) + 8*A)*(x + 1) - 4*A;
        coeffs[1] = ((A + 2)*x - (A + 3))*x*x + 1;
        coeffs[2] = ((A + 2)*(1 - x) - (A + 3))*(1 - x)*(1 - x) + 1;
        coeffs[3] = 1.f - coeffs[0] - coeffs[1] - coeffs[2];
        return coeffs;
    }
    private static int clip(int num, int min , int max) {
        return num >= min ? Math.min(num, max) : min;
    }
}