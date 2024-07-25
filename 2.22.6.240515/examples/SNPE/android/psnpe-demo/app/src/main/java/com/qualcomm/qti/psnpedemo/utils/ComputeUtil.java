/*
 * Copyright (c) 2021 Qualcomm Technologies, Inc.
 * All Rights Reserved.
 * Confidential and Proprietary - Qualcomm Technologies, Inc.
 */
package com.qualcomm.qti.psnpedemo.utils;

import android.util.Log;

import java.io.BufferedReader;

import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;

import java.io.FileReader;

import java.io.IOException;
import java.io.InputStream;
import java.nio.ByteBuffer;
import java.nio.FloatBuffer;
import java.util.ArrayList;

import java.util.List;

public class ComputeUtil {
    private static String TAG = ComputeUtil.class.getSimpleName();

    public static int[] mathFloor(float[] data) {
        if(data.length == 0) {
            return null;
        }
        int[] result = new int[data.length];
        for(int i = 0; i < data.length; i++) {
            result[i] = (int)Math.round(data[i]);
        }
        return result;
    }

    public static double mean(float[] data) {
        if(data.length == 0) {
            return Double.NaN;
        }
        double mean = 0.0;
        for(int i = 0; i < data.length; i++) {
            mean += data[i];
        }
        mean = mean / (1.0* data.length);
        return mean;
    }

    public static double mean(double[] data) {
        if(data.length == 0) {
            return Double.NaN;
        }
        double mean = 0.0;
        for(int i = 0; i < data.length; i++) {
            mean += data[i];
        }
        mean = mean / (1.0* data.length);
        return mean;
    }

    public static double mean(List<Double> data) {
        if(data.size() == 0) {
            return Double.NaN;
        }
        double mean = 0.0;
        for(int i = 0; i < data.size(); i++) {
            mean += data.get(i);
        }
        mean = mean / (1.0* data.size());
        return mean;
    }


    public static byte[] floatArrayToByteArray(float[] floats) {
        ByteBuffer buffer = ByteBuffer.allocate(4 * floats.length);
        FloatBuffer floatBuffer = buffer.asFloatBuffer();
        floatBuffer.put(floats);
        return buffer.array();
    }

    public static float[] getArray(String filename){
        try{
            byte[] a = read(filename);
            float[] b = byteArrayToFloatArray(a);
            return b;
        } catch (FileNotFoundException e) {
            Log.e(TAG, "File Not Found. " + e);
            return null;
        } catch (IOException e){
            Log.e(TAG, "IO Exception. " + e);
            return null;
        }
    }

    private static byte[] read(String fileName) throws IOException {
        InputStream is=new FileInputStream(fileName);
        ByteArrayOutputStream bos=new ByteArrayOutputStream();
        byte[] buffer=new byte[is.available()];
        int n=0;
        while((n=is.read(buffer))!=-1)
            bos.write(buffer,0,n);
        bos.close();
        is.close();
        return bos.toByteArray();
    }


    private static float[] ByteArrayToFloatArray(byte[] data){
        if(data.length == 0) {
            return null;
        }
        float[] result = new float[data.length / 4];
        int temp = 0;
        for (int i = 0; i < data.length; i += 4){
            temp = temp | (data[i] & 0xff) << 0;
            temp = temp | (data[i+1] & 0xff) << 8;
            temp = temp | (data[i+2] & 0xff) << 16;
            temp = temp | (data[i+3] & 0xff) << 24;
            result[i / 4] = Float.intBitsToFloat(temp);
            temp = 0;
        }
        return result;
    }

    public static float[] byteArrayToFloatArray(byte[] bytes) {
        ByteBuffer buffer = ByteBuffer.wrap(bytes);
        FloatBuffer fb = buffer.asFloatBuffer();
        float[] floatArray = new float[fb.limit()];
        fb.get(floatArray);
        return floatArray;
    }

    public static ArrayList readArrayFromTxt(String filePath) {
        ArrayList<String> arrayList = new ArrayList<String>();
        try{
            File file = new File(filePath);
            if(file.exists()) {
                Log.d(TAG, "file length " + file.length());
                BufferedReader br = new BufferedReader(new FileReader(filePath));
                String s;
                int i = 0;
                while ((s = br.readLine()) != null) {
                    arrayList.add(s);
                }
                br.close();
            } else {
                Log.d(TAG, "file not exist： " +  filePath);
            }

        }catch(Exception e){
            e.printStackTrace();
        }
        return arrayList;
    }



    public static Integer[][] matrixReshape(Integer[][] matrix, int height, int width) {
        if(matrix.length == 0 || matrix.length * matrix[0].length != height * width) {
            return matrix;
        }
        Integer [][] reshapedMatrix = new Integer[height][width];
        int counter = 0;
        for(int i = 0; i < matrix.length; i++) {
            for(int j = 0; j < matrix[0].length; j++) {
                reshapedMatrix[counter/width][counter%width] = matrix[i][j];
                counter++;
            }
        }
        return reshapedMatrix;
    }

    public static Integer minMatrix(Integer[][] num) {
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

    public static int searchSorted(float[] seq, double data) {
        int count = -1;
        for(int i = 0; i < seq.length; i++) {
            if(data >= seq[i]) {
                count++;
            }
        }
        if(count == -1) {
            return 0;
        }
        return count;
    }

    public static double getAverage(double[]num) {
        if(num.length == 0) {
            throw new NullPointerException();
        }
        double average = 0.0;
        for(int i = 0; i < num.length; i++) {
            average += num[i];
        }
        return (double) average/num.length;
    }

    public static Integer maxMatrix(Integer[][] num) {
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

    public static Integer[][] transposeMatrix(Integer[][] matrix) {
        Integer[][] transposedM = new Integer[matrix[0].length][matrix.length];
        for(int i = 0; i < matrix[0].length; i++) {
            for(int j = 0; j < matrix.length;j++){
                transposedM[i][j] = matrix[j][i];
            }
        }
        return transposedM;
    }

}