/*
 * Copyright (c) 2019-2020 Qualcomm Technologies, Inc.
 * All Rights Reserved.
 * Confidential and Proprietary - Qualcomm Technologies, Inc.
 */
package com.qualcomm.qti.psnpedemo.utils;

import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Color;
import android.media.Image;
import android.util.Log;
import android.view.Gravity;
import android.widget.LinearLayout;
import android.widget.TextView;
import android.widget.Toast;
import com.qualcomm.qti.psnpedemo.components.BenchmarkApplication;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.io.InputStreamReader;
import java.nio.ByteBuffer;
import java.nio.FloatBuffer;
import java.nio.IntBuffer;
import java.util.Arrays;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Map;


public class Util {
    private static String TAG = Util.class.getSimpleName();

    public static int R(int c) {
        return (c >> 16) & 0xff;
    }

    public static int G(int c) {
        return (c >> 8) & 0xff;
    }

    public static int B(int c) {
        return c & 0xff;
    }

    public static double r(int c) {
        return R(c)/255.0;
    }

    public static double g(int c) {
        return G(c)/255.0;
    }

    public static double b(int c) {
        return B(c)/255.0;
    }

    static public void writeData(String imageName, float []data) {
        File file = new File("/storage/emulated/0/Android/data/com.demo.qcbenchmark/files/" + imageName + ".txt");
        if(!file.exists()) {
            try {
                file.createNewFile();
            } catch (IOException e) {
                e.printStackTrace();
            }
        }
        try {
            FileWriter fileWriter = new FileWriter(file, false);
            BufferedWriter bufferedWriter = new BufferedWriter(fileWriter);
            StringBuilder sb = new StringBuilder();
            for(int i=0; i<data.length; i++) {
                sb.append(data[i]).append("\n");
            }
            sb.deleteCharAt(sb.length()-1);
            //Log.d("Utils", sb.toString());
            bufferedWriter.write(sb.toString());
            bufferedWriter.close();

        } catch (Exception e) {
            e.printStackTrace();
        }

    }

    public static float[] imagePreprocess(File imageName, int inputsize, double[] meanRGB, double div, boolean useBGR, int resize_target) {
        float[] pixelFloats = new float[inputsize * inputsize * 3];
        Bitmap img;
        Bitmap imgcrop;
        Bitmap imgresize;
        try{
            img = BitmapFactory.decodeStream(new FileInputStream(imageName));
            //resize image
            int short_dim = Math.min(img.getHeight(), img.getWidth());
            if(resize_target == 300) {
                imgresize = img;
                imgcrop = Bitmap.createScaledBitmap(imgresize, inputsize, inputsize, true);//resize
            }else {
                int px = (img.getWidth() - short_dim) / 2;
                int py = (img.getHeight() - short_dim) / 2;
                imgresize = Bitmap.createBitmap(img, px, py, short_dim, short_dim, null, false);//crop
                imgcrop = Bitmap.createScaledBitmap(imgresize, inputsize, inputsize, true);//resize
            }
            final int[] pixels = new int[imgcrop.getWidth() * imgcrop.getHeight()];
            imgcrop.getPixels(pixels, 0, imgcrop.getWidth(), 0, 0,
                    imgcrop.getWidth(), imgcrop.getHeight());
            int z = 0;
            for (int y = 0; y < imgcrop.getHeight(); y++) {
                for (int x = 0; x < imgcrop.getWidth(); x++) {
                    final int rgb = pixels[y * imgcrop.getWidth() + x];
                    float b = (((rgb) & 0xFF) - (float) meanRGB[2]) / (float)div;
                    float g = (((rgb >> 8) & 0xFF) - (float) meanRGB[1]) / (float)div;
                    float r = (((rgb >> 16) & 0xFF) - (float) meanRGB[0]) / (float)div;
                    if (useBGR) {
                        pixelFloats[z++] = b;
                        pixelFloats[z++] = g;
                        pixelFloats[z++] = r;
                    }
                    else {
                        pixelFloats[z++] = r;
                        pixelFloats[z++] = g;
                        pixelFloats[z++] = b;
                    }
                }
            }
            img.recycle();
            imgcrop.recycle();
            imgresize.recycle();
        } catch (IOException e) {
            e.printStackTrace();
        }
        return pixelFloats;
    }

    public static void write2file(String inputName, String content) {
        File file = new File(inputName);
        if(!file.exists()) {
            try {
                file.createNewFile();
            } catch (IOException e) {
                e.printStackTrace();
            }
        }
        try {
            FileWriter fileWriter = new FileWriter(file, true);
            BufferedWriter bufferedWriter = new BufferedWriter(fileWriter);
            bufferedWriter.write(content+"\n");
            bufferedWriter.close();
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    public static void writeData(String imageName, double []data) {
        File file = new File("/storage/emulated/0/Android/data/com.demo.qcbenchmark/files/" + imageName + ".txt");
        if(!file.exists()) {
            try {
                file.createNewFile();
            } catch (IOException e) {
                e.printStackTrace();
            }
        }
        try {
            FileWriter fileWriter = new FileWriter(file, false);
            BufferedWriter bufferedWriter = new BufferedWriter(fileWriter);
            StringBuilder sb = new StringBuilder();
            for(int i=0; i<data.length; i++) {
                sb.append(data[i]).append("\n");
            }
            sb.deleteCharAt(sb.length()-1);
            //Log.d("Utils", sb.toString());
            bufferedWriter.write(sb.toString());
            bufferedWriter.close();

        } catch (Exception e) {
            e.printStackTrace();
        }

    }

    public static void writeTopK(String imageName, HashMap<String, Boolean> data) {
        File file = new File("/storage/emulated/0/Android/data/com.demo.qcbenchmark/files/" + imageName + ".txt");
        if(!file.exists()) {
            try {
                file.createNewFile();
            } catch (IOException e) {
                e.printStackTrace();
            }
        }
        try {
            FileWriter fileWriter = new FileWriter(file, false);
            BufferedWriter bufferedWriter = new BufferedWriter(fileWriter);
            StringBuilder sb = new StringBuilder();
            for(Map.Entry<String, Boolean> entry : data.entrySet()) {
                sb.append(entry.getKey()).append("   " + entry.getValue() + "\n");
            }
            sb.deleteCharAt(sb.length()-1);
            //Log.d("Utils", sb.toString());
            bufferedWriter.write(sb.toString());
            bufferedWriter.close();

        } catch (Exception e) {
            e.printStackTrace();
        }

    }

    public static void displayAtToast(Context context, String str, int showTime) {
        Toast toast = Toast.makeText(context, str, showTime);
        toast.setGravity(Gravity.CENTER , 0, 0);  //display position
        LinearLayout layout = (LinearLayout) toast.getView();
        layout.setBackgroundColor(Color.GRAY);
        TextView v = toast.getView().findViewById(android.R.id.message);
        v.setTextColor(Color.RED);     //font color
        toast.show();
    }

    public static boolean deleteFolderFile(File path) {
        if (path.isDirectory()) {
            if (path.listFiles().length != 0) {
                for (File file : path.listFiles()) {
                    boolean res = deleteFolderFile(file);
                    if (!res) {
                        return false;
                    }
                }
            }
            return true;
        } else {
            return path.delete();
        }
    }

    public static boolean delete(File file) {
        if (!file.exists()){
            return true;
        }
        if (file.isDirectory()) {
            File[] filelist = file.listFiles();
            if (null != filelist){
                for (File subfile : filelist){
                    delete(subfile);
                }
            }
        }
        return file.delete();
    }

    public static float[] readArrayFromTxt(String filePath) {
        float [] result = null;
        try{
            File file = new File(filePath);
            if(file.exists()) {
                result = new float[(int) file.length()];
                Log.d(TAG, "file length " + file.length());
                BufferedReader br = new BufferedReader(new FileReader(filePath));//构造一个BufferedReader类来读取文件
                String s;
                int i = 0;
                while ((s = br.readLine()) != null) {//使用readLine方法，一次读一行
                    result[i++] = Float.parseFloat(s);
                }
                br.close();
            } else {
                Log.d(TAG, "file not exist： " +  filePath);
            }

        }catch(Exception e){
            e.printStackTrace();
        }
        return result;
    }

    public static int checkImageDirValidation(String imagePath) {
        // check if imagePath contains images
        File images = new File(imagePath);
        if(!images.exists() ) {
            Log.e(TAG, "Image path not exists:" + imagePath);
            return 0;
        }
        else if (0 == images.listFiles().length){
            Log.w(TAG, "No data exist in the paht:" + imagePath);
        }
        return images.listFiles().length;
    }

    public static void clearInputList(String modelName) {
        String inputPath = BenchmarkApplication.getCustomApplicationContext().getExternalFilesDir("input_list").getAbsolutePath();
        if(modelName.contains("deeplab"))
            inputPath = inputPath + "/deeplabv3_input_list.txt";
        else if(modelName.contains("ssd"))
            inputPath = inputPath + "/mobilenetssd_input_list.txt";
        else if(modelName.contains("vdsr"))
            inputPath = inputPath + "/vdsr_input_list.txt";

        File file = new File(inputPath);
        if(file.exists())
            file.delete();
    }

    public static void writeArrayTofile(String filePath, float[] data, boolean makedirs){
        try {
            if (makedirs){
                int dirPathEnd = filePath.lastIndexOf('/');
                if (-1 != dirPathEnd){
                    String dirPath = filePath.substring(0, dirPathEnd);
                    File dir = new File(dirPath);
                    if (!dir.exists()){
                        if (!dir.mkdirs()){
                            Log.e(TAG, String.format("Create dir \"%s\" fail.", dirPath));
                        }
                    }
                    else if (!dir.isDirectory()){
                        Log.e(TAG, String.format("Create dir \"%s\" fail. It's a file.", dirPath));
                    }
                }
            }
            FileOutputStream os = new FileOutputStream(filePath);
            float[] buf = new float[data.length];
            for (int i = 0; i < data.length; ++i) {
                int intBits = Float.floatToIntBits(data[i]);
                intBits = Integer.reverseBytes(intBits);
                buf[i] = Float.intBitsToFloat(intBits);
            }
            ByteBuffer byteBuf = ByteBuffer.allocate(4 * data.length);
            FloatBuffer floatBuf = byteBuf.asFloatBuffer();
            floatBuf.put(buf);
            os.write(byteBuf.array());
            os.close();
        }
        catch (Exception e){
            Log.e(TAG, e.toString());
            e.printStackTrace();
        }
    }

    public static float[] readFloatArrayFromFile(File file){
        try {
            FileInputStream is = new FileInputStream(file);
            byte[] bytes = new byte[is.available()];
            is.read(bytes);
            float[] floatArray = new float[bytes.length / 4];
            for (int i = 0; i < bytes.length; i += 4) {
                floatArray[i/4] = Float.intBitsToFloat((0xff & bytes[i]) | (0xff00 & (bytes[i+1] << 8))
                        | (0xff0000 & (bytes[i+2] << 16)) | (0xff000000 & (bytes[i+3] << 24)));
            }
            return floatArray;
        }
        catch (Exception e){
            Log.e(TAG, e.toString());
            e.printStackTrace();
        }
        return null;
    }

    public static float[] readFloatArrayFromFile(String filePath){
        try {
            return readFloatArrayFromFile(new File(filePath));
        }
        catch (Exception e){
            Log.e(TAG, e.toString());
            e.printStackTrace();
        }
        return null;
    }

    public static  int[][] readImageToPmode(String imgPath) {
        if(imgPath == "") {
            return null;
        }
        int[][] image = null;
        Bitmap img = null;
        HashMap<String, Integer> pmodeToRgbMap = new HashMap<String, Integer>();
        String[] rgbKeyList={"[0, 0, 0]",
                "[128, 0, 0]",
                "[0, 128, 0]",
                "[128, 128, 0]",
                "[0, 0, 128]",
                "[128, 0, 128]",
                "[0, 128, 128]",
                "[128, 128, 128]",
                "[64, 0, 0]",
                "[192, 0, 0]",
                "[64, 128, 0]",
                "[192, 128, 0]",
                "[64, 0, 128]",
                "[192, 0, 128]",
                "[64, 128, 128]",
                "[192, 128, 128]",
                "[0, 64, 0]",
                "[128, 64, 0]",
                "[0, 192, 0]",
                "[128, 192, 0]",
                "[0, 64, 128]"};
        for(int index=0;index<21;index++){
            pmodeToRgbMap.put(rgbKeyList[index],index);
        }
        pmodeToRgbMap.put("[224, 224, 192]",255);
        try {
            img = BitmapFactory.decodeStream(new FileInputStream(imgPath));
            int height = img.getHeight();
            int width = img.getWidth();
            image = new int[height][width];
            int pixel,  r, g, b;
            for(int j = 0; j < height; j++)
            {
                for(int i = 0; i < width; i++)
                {
                    pixel = img.getPixel(i,j);
                    r=Color.red(pixel);
                    g= Color.green(pixel);
                    b= Color.blue(pixel);
                    int[] rgbarray={r,g,b};
                    String rgbKey= Arrays.toString(rgbarray);
                    image[j][i] = pmodeToRgbMap.get(rgbKey);
                }
            }
        } catch (FileNotFoundException e) {
            Log.e(TAG, "File not found: " + imgPath);
        } finally {
            if(img != null)
                img.recycle();
        }
        return image;
    }

    public static int[][] getResizedResultImage(float[] resultVec,  int height, int width,int FIXED_HEIGHT, int FIXED_WIDTH) {
        if(resultVec == null) {
            Log.e(TAG, "Result vec is null.");
            return null;
        }

        int[][] result;

        if(width * height == resultVec.length) {
            result = MathUtils.matrixReshape(resultVec, height, width);
        }
        else{
            result = MathUtils.matrixReshapeAndCut(resultVec, height, width, FIXED_HEIGHT, FIXED_WIDTH);
        }

        return result;

    }
    public static String getDatafromFile(String fileName) {
        BufferedReader reader = null;
        String laststr = "";
        try {
            FileInputStream fileInputStream = new FileInputStream(fileName);
            InputStreamReader inputStreamReader = new InputStreamReader(fileInputStream, "UTF-8");
            reader = new BufferedReader(inputStreamReader);
            String tempString = null;
            while ((tempString = reader.readLine()) != null) {
                laststr += tempString;
            }
            reader.close();
        } catch (FileNotFoundException e){
            Log.e(TAG, "file not found. " + fileName + e);
        }
        catch (IOException e) {
            Log.e(TAG, "IO Exception.");
        } finally {
            if (reader != null) {
                try {
                    reader.close();
                } catch (IOException e) {
                    Log.e(TAG, "reader close Exception.");
                }
            }
        }
        return laststr;
    }

    public static File[] readImageList(String path){
        try {
            File listFile = new File(path);
            BufferedReader bufReader = new BufferedReader(new FileReader(listFile));
            ArrayList<String> imgPathList = new ArrayList<>();
            String line = bufReader.readLine();
            while (null != line){
                if(line.equals("")){
                    break;
                }
                imgPathList.add(listFile.getParent() + "/" + line);
                line = bufReader.readLine();
            }
            File[] imageList = new File[imgPathList.size()];
            for (int i = 0; i < imageList.length; ++i) {
                imageList[i] = new File(imgPathList.get(i));
            }
            return imageList;
        }
        catch (Exception e){
            Log.w(TAG, e.getMessage());
        }
        return null;
    }
}