/*
 * Copyright (c) 2016-2021, 2023 Qualcomm Technologies, Inc.
 * All Rights Reserved.
 * Confidential and Proprietary - Qualcomm Technologies, Inc.
 */
package com.qualcomm.qti.snpe.imageclassifiers.tasks;

import android.app.Application;
import android.content.Context;
import android.os.AsyncTask;
import android.util.Log;

import com.qualcomm.qti.snpe.SNPE;
import com.qualcomm.qti.snpe.imageclassifiers.Model;
import com.qualcomm.qti.snpe.imageclassifiers.ModelCatalogueFragmentController;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileFilter;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.LinkedHashSet;
import java.util.LinkedList;
import java.util.List;
import java.util.Set;
import java.util.concurrent.TimeUnit;

public class LoadModelsTask extends AsyncTask<Void, Void, Set<Model>> {

    public static final String MODEL_DLC_FILE_NAME = "model.dlc";
    public static final String MODEL_MEAN_IMAGE_FILE_NAME = "mean_image.bin";
    public static final String LABELS_FILE_NAME = "labels.txt";
    public static final String IMAGES_FOLDER_NAME = "images";
    public static final String RAW_EXT = ".raw";
    public static final String UDO_PACKAGE_EXT = "Reg.so";
    public static final String UDO_PACKAGE_DIR = "/udo/arm64-v8a";
    public static final String JPG_EXT = ".jpg";
    private static final String LOG_TAG = LoadModelsTask.class.getSimpleName();

    private final ModelCatalogueFragmentController mController;

    private final Context mContext;

    public LoadModelsTask(Context context, ModelCatalogueFragmentController controller) {
        mContext = context.getApplicationContext();
        mController = controller;
    }

    @Override
    protected Set<Model> doInBackground(Void... params) {
        final Set<Model> result = new LinkedHashSet<>();
        final File modelsRoot = mContext.getExternalFilesDir("models");
        if (modelsRoot != null) {
            result.addAll(createModels(modelsRoot));
        }
        return result;
    }

    @Override
    protected void onPostExecute(Set<Model> models) {
        mController.onModelsLoaded(models);
    }

    private Set<Model> createModels(File modelsRoot) {
        final Set<Model> models = new LinkedHashSet<>();
        final Set<String> availableModels = mController.getAvailableModels();
        for (File child : modelsRoot.listFiles()) {
            if (!child.isDirectory() || !availableModels.contains(child.getName())) {
                continue;
            }
            try {
                models.add(createModel(child));
            } catch (IOException e) {
                Log.e(LOG_TAG, "Failed to load model from model directory.", e);
            }
        }
        return models;
    }

    private Model createModel(File modelDir) throws IOException {
        final Model model = new Model();
        model.name = modelDir.getName();
        model.file = new File(modelDir, MODEL_DLC_FILE_NAME);
        model.meanImage = new File(modelDir, MODEL_MEAN_IMAGE_FILE_NAME);
        final File images = new File(modelDir, IMAGES_FOLDER_NAME);

        // Paths to UDO libs
        String udoDir = mContext.getFilesDir().getPath() + "/" + model.name + "/udo/";
        model.udoDir = new File(udoDir);

        FileFilter udoFilter = new FileFilter() {
                @Override
                public boolean accept(File file) {
                    return file.getName().endsWith(UDO_PACKAGE_EXT);
                }
            };

        model.udoConfigs = model.udoDir.listFiles(udoFilter);
        if (model.udoConfigs != null && model.udoConfigs.length == 0) {
            try{
                TimeUnit.MICROSECONDS.sleep(500);
            }
            catch(Exception e){
                System.out.println(e);
            }
            model.udoConfigs = model.udoDir.listFiles(udoFilter);
        }

        String udoDirPath = modelDir.getAbsolutePath() + UDO_PACKAGE_DIR;
        model.udoArmDir = new File(udoDirPath);
        udoRegister(model);

        if (images.isDirectory()) {
            model.rawImages = images.listFiles(new FileFilter() {
                @Override
                public boolean accept(File file) {
                    return file.getName().endsWith(RAW_EXT);
                }
            });
            model.jpgImages = images.listFiles(new FileFilter() {
                @Override
                public boolean accept(File file) {
                    return file.getName().endsWith(JPG_EXT);
                }
            });
        }
        model.labels = loadLabels(new File(modelDir, LABELS_FILE_NAME));
        return model;
    }

    private String[] loadLabels(File labelsFile) throws IOException {
        final List<String> list = new LinkedList<>();
        final BufferedReader inputStream = new BufferedReader(
            new InputStreamReader(new FileInputStream(labelsFile)));
        String line;
        while ((line = inputStream.readLine()) != null) {
            list.add(line);
        }
        return list.toArray(new String[list.size()]);
    }

    private void udoRegister(Model mModel){
        if(mModel.udoConfigs != null){
            for (final File file : mModel.udoArmDir.listFiles()) {
                String udoArmLibs = file.getName();
                String udoLibName = mModel.udoDir.getAbsolutePath() + "/" + udoArmLibs;
                System.load(udoLibName);
            }

            for (final File file : mModel.udoConfigs) {
                SNPE.addOpPackage((Application) mContext, file.getAbsolutePath());
            }

        }
    }
}
