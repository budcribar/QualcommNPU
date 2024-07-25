/*
 * Copyright (c) 2016-2018 Qualcomm Technologies, Inc.
 * All Rights Reserved.
 * Confidential and Proprietary - Qualcomm Technologies, Inc.
 */
package com.qualcomm.qti.snpe.imageclassifiers.tasks;

import android.graphics.Bitmap;
import android.os.AsyncTask;
import android.util.Pair;

import com.qualcomm.qti.snpe.NeuralNetwork;
import com.qualcomm.qti.snpe.imageclassifiers.Model;
import com.qualcomm.qti.snpe.imageclassifiers.ModelOverviewFragmentController;

import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;
import java.util.Set;

public abstract class AbstractClassifyImageTask extends AsyncTask<Bitmap, Void, String[]> {

    private static final String LOG_TAG = AbstractClassifyImageTask.class.getSimpleName();

    private static final int FLOAT_SIZE = 4;

    final String mInputLayer;

    final String mOutputLayer;

    private final ModelOverviewFragmentController mController;

    final NeuralNetwork mNeuralNetwork;

    final Model mModel;

    final Bitmap mImage;

    private FloatBuffer mMeanImage;

    long mJavaExecuteTime = -1;

    AbstractClassifyImageTask(ModelOverviewFragmentController controller,
                                     NeuralNetwork network, Bitmap image, Model model) {
        mController = controller;
        mNeuralNetwork = network;
        mImage = image;
        mModel = model;

        Set<String> inputNames = mNeuralNetwork.getInputTensorsNames();
        Set<String> outputNames = mNeuralNetwork.getOutputTensorsNames();
        if (inputNames.size() != 1 || outputNames.size() != 1) {
            throw new IllegalStateException("Invalid network input and/or output tensors.");
        } else {
            mInputLayer = inputNames.iterator().next();
            mOutputLayer = outputNames.iterator().next();
        }

    }

    @Override
    protected void onPostExecute(String[] labels) {
        super.onPostExecute(labels);
        if (labels.length > 0) {
            mController.onClassificationResult(labels, mJavaExecuteTime);
        } else {
            mController.onClassificationFailed();
        }
    }

    void loadMeanImageIfAvailable(File meanImage, final int imageSize) {
        if (!meanImage.exists()) {
            return;
        }
        FileInputStream fileInputStream = null;
        try {
            fileInputStream = new FileInputStream(meanImage);
            if (fileInputStream == null)
            {
               return;
            }
            ByteBuffer buffer = ByteBuffer.allocateDirect(imageSize * FLOAT_SIZE)
                .order(ByteOrder.nativeOrder());
            final byte[] chunk = new byte[1024];
            int read;
            while ((read = fileInputStream.read(chunk)) != -1) {
                buffer.put(chunk, 0, read);
            }
            buffer.flip();
            mMeanImage = buffer.asFloatBuffer();
            if (fileInputStream != null) {
               fileInputStream.close();
            }
        } catch (IOException e) {}
    }
    float[] loadRgbBitmapAsFloat(Bitmap image) {
        final float[] pixels = new float[image.getWidth() * image.getHeight() * 3];
        for (int y = image.getHeight() - 1; y >= 0; y--) {
            for (int x = image.getWidth() - 1; x >= 0; x--) {
                final int idx = y * image.getWidth() + x;
                final int batchIdx = idx * 3;
                int pixel = image.getPixel(x, y);
                String modelName = mModel.name;
                if (modelName.equals("inception_v3")) {
                   float grayscale = ((pixel >> 16) & 0xFF);
                   pixels[batchIdx] = (grayscale - 128) / 128;
                   grayscale = ((pixel >>  8) & 0xFF);
                   pixels[batchIdx + 1] = (grayscale - 128) / 128;
                   grayscale = (pixel & 0xFF);
                   pixels[batchIdx + 2] = (grayscale - 128) / 128;
                } else if (modelName.equals("alexnet") && mMeanImage != null) {
                   float grayscale = ((pixel >> 16) & 0xFF);
                   pixels[batchIdx] = grayscale - mMeanImage.get();
                   grayscale = ((pixel >>  8) & 0xFF);
                   pixels[batchIdx + 1] = grayscale - mMeanImage.get();
                   grayscale = (pixel & 0xFF);
                   pixels[batchIdx + 2] = grayscale - mMeanImage.get();
                } else if (modelName.equals("googlenet") && mMeanImage != null) {
                   float grayscale = ((pixel >> 16) & 0xFF);
                   pixels[batchIdx] = grayscale - mMeanImage.get();
                   grayscale = ((pixel >>  8) & 0xFF);
                   pixels[batchIdx + 1] = grayscale - mMeanImage.get();
                   grayscale = (pixel & 0xFF);
                   pixels[batchIdx + 2] = grayscale - mMeanImage.get();
                }
                else {
                   float grayscale = ((pixel >> 16) & 0xFF);
                   pixels[batchIdx] = grayscale;
                   grayscale = ((pixel >>  8) & 0xFF);
                   pixels[batchIdx + 1] = grayscale;
                   grayscale = (pixel & 0xFF);
                   pixels[batchIdx + 2] = grayscale;
                }
            }
        }
        return pixels;
    }

    float[] loadGrayScaleBitmapAsFloat(Bitmap image) {
        float[] pixels = new float[image.getWidth() * image.getHeight()];
        for (int y = 0; y < image.getHeight(); y++) {
            for (int x = 0; x < image.getWidth(); x++) {
                final int idx = y * image.getWidth() + x;
                final int rgb = image.getPixel(x, y);
                final float b = ((rgb)       & 0xFF);
                final float g = ((rgb >>  8) & 0xFF);
                final float r = ((rgb >> 16) & 0xFF);
                float grayscale = (float) (r * 0.3 + g * 0.59 + b * 0.11);
                pixels[idx] = grayscale;
                String modelName = mModel.name;
                if (modelName.equals("inception_v3")) {
                   pixels[idx] = (grayscale - 128) / 128;
                } else if (modelName.equals("alexnet") && mMeanImage != null) {
                   pixels[idx] = grayscale - mMeanImage.get();
                } else if (modelName.equals("googlenet") && mMeanImage != null) {
                   pixels[idx] = grayscale - mMeanImage.get();
                }
            }
        }
        return pixels;
    }

    Pair<Integer, Float>[] topK(int k, final float[] tensor) {
        final boolean[] selected = new boolean[tensor.length];
        final Pair<Integer, Float> topK[] = new Pair[k];
        int count = 0;
        while (count < k) {
            final int index = top(tensor, selected);
            selected[index] = true;
            topK[count] = new Pair<>(index, tensor[index]);
            count++;
        }
        return topK;
    }

    private int top(final float[] array, boolean[] selected) {
        int index = 0;
        float max = -1.f;
        for (int i = 0; i < array.length; i++) {
            if (selected[i]) {
                continue;
            }
            if (array[i] > max) {
                max = array[i];
                index = i;
            }
        }
        return index;
    }

    private float preProcess(float original) {
        String modelName = mModel.name;

        if (modelName.equals("inception_v3")) {
            return (original - 128) / 128;
        } else if (modelName.equals("alexnet") && mMeanImage != null) {
            return original - mMeanImage.get();
        } else if (modelName.equals("googlenet") && mMeanImage != null) {
            return original - mMeanImage.get();
        } else {
            return original;
        }
    }

    float getMin(float[] array) {
        float min = Float.MAX_VALUE;
        for (float value : array) {
            if (value < min) {
                min = value;
            }
        }
        return min;
    }

    float getMax(float[] array) {
        float max = Float.MIN_VALUE;
        for (float value : array) {
            if (value > max) {
                max = value;
            }
        }
        return max;
    }
}
