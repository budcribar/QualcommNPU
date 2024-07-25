/*
 * Copyright (c) 2019 Qualcomm Technologies, Inc.
 * All Rights Reserved.
 * Confidential and Proprietary - Qualcomm Technologies, Inc.
 */
package com.qualcomm.qti.psnpedemo.views;

import android.app.Activity;
import android.content.Intent;
import android.os.Bundle;
import android.util.Log;
import android.view.View;
import android.widget.AdapterView;
import android.widget.ListView;
import android.widget.TextView;
import android.widget.Toast;


import com.qualcomm.qti.psnpe.PSNPEConfig;
import com.qualcomm.qti.psnpe.PSNPEManager;
import com.qualcomm.qti.psnpedemo.R;
import com.qualcomm.qti.psnpedemo.components.ModelItemAdapter;
import com.qualcomm.qti.psnpedemo.components.ModelListItem;
import com.qualcomm.qti.psnpedemo.utils.Util;
import java.io.File;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;

public class MainActivity extends Activity {
    private static String TAG = MainActivity.class.getSimpleName();
    TextView message;
    ListView modelList;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        message = findViewById(R.id.messageText);
        message.setVisibility(View.GONE);
        String nativeLibPath = getApplicationInfo().nativeLibraryDir;
        String runtimeConfigPath = getExternalFilesDir("configs").getAbsolutePath();
        if(!PSNPEManager.init(nativeLibPath,  runtimeConfigPath + "/model_configs.json")) {
            Util.displayAtToast(MainActivity.this, "PSNPE init failed, natviePath: " + nativeLibPath + " or model configs file not right", Toast.LENGTH_LONG);
            return;
        }
        initUI();
    }

    private boolean loadModels() {
        Map<String, PSNPEConfig> modelConfigs = PSNPEManager.getPSNPEConfig();
        List<ModelListItem> models = new LinkedList<>();
        for(Map.Entry<String, PSNPEConfig>  configEntry : modelConfigs.entrySet()) {
            String modelName = configEntry.getKey();
            PSNPEConfig config = configEntry.getValue();
            String scenario;

            if(!(new File(config.modelFile).exists()))
                continue;

            if (config.modelFile.contains("superresolution")) {
                scenario = "superresolution";
            } else if(config.modelFile.contains("detection")) {
                scenario = "detection";
            } else if(config.modelFile.contains("segmentation")) {
                scenario = "segmentation";
            } else if(config.modelFile.contains("naturallanguage")) {
                scenario = "naturallanguage";
            }else if(config.modelFile.contains("classification")) {
                scenario = "classification";
            } else if(config.modelFile.contains("FaceRecognition")) {
                scenario = "FaceRecognition";
            } else {
                return false;
            }

            ModelListItem item = new ModelListItem(scenario, modelName);
            models.add(item);
        }

        if(models.size() > 0) {
            modelList.setAdapter(new ModelItemAdapter(MainActivity.this,
                    R.layout.listview_item, models));
            modelList.setOnItemClickListener(new AdapterView.OnItemClickListener() {
                @Override
                public void onItemClick(AdapterView<?> adapterView, View view, int i, long l) {
                    ModelListItem modelListItem = (ModelListItem) modelList.getItemAtPosition(i);
                    Intent intent = new Intent();
                    intent.setClass(MainActivity.this, modelActivity.class);
                    Bundle bundle = new Bundle();
                    bundle.putString("scenario", modelListItem.getScenarioName());
                    bundle.putString("model", modelListItem.getModelName());
                    intent.putExtra("key", bundle);
                    startActivity(intent);
                }
            });
            return true;
        }

        return false;
    }

    public void initUI() {
        modelList = findViewById(R.id.scenarioList);

        initDir();
        if(!loadModels()) {
            Log.d("MainActivity","No Model exist, please check your model config file");
            message.setVisibility(View.VISIBLE);
            message.setText("No Model exist, please check your model config file");
            return ;
        }
    }

    void initDir() {
        File inputListDir = getExternalFilesDir("input_list");
        File resultDir = getExternalFilesDir("results");
        if(!Util.deleteFolderFile(inputListDir))
            Log.d(TAG, "Delete files from input_list failed");
        if(!Util.deleteFolderFile(resultDir))
            Log.d(TAG, "Delete files from results failed");
    }
}
