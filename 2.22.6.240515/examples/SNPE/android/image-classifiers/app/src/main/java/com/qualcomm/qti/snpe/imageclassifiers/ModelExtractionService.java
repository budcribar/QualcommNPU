/*
 * Copyright (c) 2016-2021, 2023 Qualcomm Technologies, Inc.
 * All Rights Reserved.
 * Confidential and Proprietary - Qualcomm Technologies, Inc.
 */
package com.qualcomm.qti.snpe.imageclassifiers;

import android.app.IntentService;
import android.content.Intent;
import android.content.Context;
import android.net.Uri;
import android.util.Log;

import java.io.File;
import java.io.FileOutputStream;
import java.io.FileInputStream;
import java.io.InputStream;
import java.io.OutputStream;
import java.io.IOException;
import java.util.zip.ZipEntry;
import java.util.zip.ZipInputStream;

public class ModelExtractionService extends IntentService {

    private static final String LOG_TAG = ModelExtractionService.class.getSimpleName();
    private static final String ACTION_EXTRACT = "extract";
    private static final String EXTRA_MODEL_RAW_RES_ID = "model_raw_res";
    private static final String EXTRA_MODEL_NAME = "model_name";
    public static final String MODELS_ROOT_DIR = "models";
    private static final int CHUNK_SIZE = 1024;
    private static Context mContext;

    public ModelExtractionService() {
        super("ModelExtractionService");
    }

    public static void extractModel(final Context context, final String modelName,
        final int modelRawResId) {
        mContext = context;
        Intent intent = new Intent(context, ModelExtractionService.class);
        intent.setAction(ACTION_EXTRACT);
        intent.putExtra(EXTRA_MODEL_NAME, modelName);
        intent.putExtra(EXTRA_MODEL_RAW_RES_ID, modelRawResId);
        context.startService(intent);
    }

    @Override
    protected void onHandleIntent(Intent intent) {
        if (intent != null) {
            final String action = intent.getAction();
            if (ACTION_EXTRACT.equals(action)) {
                final int modelRawResId = intent.getIntExtra(EXTRA_MODEL_RAW_RES_ID, 0);
                final String modelName = intent.getStringExtra(EXTRA_MODEL_NAME);
                handleModelExtraction(modelName, modelRawResId);
            }
        }
    }

    private void extractUdoConfig(String modelDirName, String udoFolderName){
        final File udoConfigDir = new File(modelDirName + udoFolderName);
            if(udoConfigDir.exists()){
                try{
                    File modelDir = new File(modelDirName);
                    for (final File file : udoConfigDir.listFiles()) {
                        InputStream is = null;
                        OutputStream os = null;
                        String destName = mContext.getFilesDir().getPath();
                        destName = destName + "/" + modelDir.getName() + "/udo/";
                        File destDir = new File(destName);
                        destDir.mkdirs();
                        File destination = new File(destName + file.getName());
                        try {
                            byte[] buffer = new byte[1024];
                            int length;
                            is = new FileInputStream(file);
                            os = new FileOutputStream(destination);
                            while ((length = is.read(buffer)) > 0) {
                                os.write(buffer, 0, length);
                            }
                        } finally {
                            is.close();
                            os.close();
                        }
                    }
                }
                catch(IOException e){
                    Log.e(LOG_TAG, "Failed to get Udo configs");
                }
            }
    }

    private void handleModelExtraction(final String modelName, final int modelRawResId) {
        ZipInputStream zipInputStream = null;
        try {
            final File modelsRoot = getOrCreateExternalModelsRootDirectory();
            final File modelRoot = createModelDirectory(modelsRoot, modelName);
            if (modelExists(modelRoot)) {
                return;
            }

            zipInputStream = new ZipInputStream(getResources().openRawResource(modelRawResId));
            ZipEntry zipEntry;
            while ((zipEntry = zipInputStream.getNextEntry()) != null) {
                final File entry = new File(modelRoot, zipEntry.getName());
                if (zipEntry.isDirectory()) {
                    doCreateDirectory(entry);
                } else {
                    doCreateFile(entry, zipInputStream);
                }
                zipInputStream.closeEntry();
            }
            getContentResolver().notifyChange(FileProvider.getFileFromUri(
                Uri.withAppendedPath(Model.MODELS_URI, modelName)), null);
            //Copy UDO configs to context directory.
            extractUdoConfig(modelRoot.getAbsolutePath(), "/udo/arm64-v8a/");
            extractUdoConfig(modelRoot.getAbsolutePath(), "/udo/dsp/");
        } catch (IOException e) {
            Log.e(LOG_TAG, e.getMessage(), e);
            try {
                if (zipInputStream != null) {
                    zipInputStream.close();
                }
            } catch (IOException ignored) {}
            getContentResolver().notifyChange(FileProvider.getFileFromUri(Model.MODELS_URI), null);
        }

    }

    private boolean modelExists(File modelRoot) {
        return modelRoot.listFiles().length > 0;
    }

    private void doCreateFile(File file, ZipInputStream inputStream) throws IOException {
        final FileOutputStream outputStream = new FileOutputStream(file);
        final byte[] chunk = new byte[CHUNK_SIZE];
        int read;
        while ((read = inputStream.read(chunk)) != -1) {
            outputStream.write(chunk, 0, read);
        }
        outputStream.close();
    }

    private void doCreateDirectory(File directory) throws IOException {
        if (!directory.mkdirs()) {
            throw new IOException("Can not create directory: " + directory.getAbsolutePath());
        }
    }

    private File getOrCreateExternalModelsRootDirectory() throws IOException {
        final File modelsRoot = getExternalFilesDir(MODELS_ROOT_DIR);
        if (modelsRoot == null) {
            throw new IOException("Unable to access application external storage.");
        }

        if (!modelsRoot.isDirectory() && !modelsRoot.mkdir()) {
            throw new IOException("Unable to create model root directory: " +
                modelsRoot.getAbsolutePath());
        }
        return modelsRoot;
    }

    private File createModelDirectory(File modelsRoot, String modelName) throws IOException {
        final File modelRoot = new File(modelsRoot, modelName);
        if (!modelRoot.isDirectory() && !modelRoot.mkdir()) {
            throw new IOException("Unable to create model root directory: " +
                modelRoot.getAbsolutePath());
        }
        return modelRoot;
    }

}
