/*
 * Copyright (c) 2019-2020 Qualcomm Technologies, Inc.
 * All Rights Reserved.
 * Confidential and Proprietary - Qualcomm Technologies, Inc.
 */
package com.qualcomm.qti.psnpedemo.views;

import android.app.Activity;
import android.content.Intent;
import android.graphics.Color;
import android.os.AsyncTask;
import android.os.Bundle;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.TextView;
import android.widget.Toast;
import com.qualcomm.qti.psnpe.PSNPEConfig;
import com.qualcomm.qti.psnpe.PSNPEManager;
import com.qualcomm.qti.psnpedemo.R;
import com.qualcomm.qti.psnpedemo.networkEvaluation.EvaluationCallBacks;
import com.qualcomm.qti.psnpedemo.networkEvaluation.ModelInfo;
import com.qualcomm.qti.psnpedemo.networkEvaluation.NetworkEvaluator;
import com.qualcomm.qti.psnpedemo.utils.Util;

import java.lang.ref.WeakReference;
import java.util.Map;

public class modelActivity extends Activity {
    private static String TAG = modelActivity.class.getSimpleName();
    // network
    NetworkEvaluator networkEvaluator;
    // views
    Button showResultButton, runButton, stressButton;
    TextView testInfo, executeStatus;
    String modelName, scenarioName;
    boolean stressTestRunning = false;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.model_main);
        initView();
    }

    void initView() {
        Intent intent = getIntent();
        Bundle bundle = intent.getBundleExtra("key");
        modelName = bundle.getString("model");
        scenarioName = bundle.getString("scenario");
        executeStatus = findViewById(R.id.executeStatus);
        showResultButton = findViewById(R.id.result_button);
        showResultButton.setClickable(false);
        showResultButton.setTextColor(Color.GRAY);
        runButton = findViewById(R.id.run_button);
        stressButton = findViewById(R.id.stress_run_button);
        testInfo = findViewById(R.id.testInfo);
        // initialized networkEvaluator
        initNetworkEvaluator();


        // set progressbar, bulk amount max number. Image num should be obtained after initModelInfo()
        Log.i(TAG, networkEvaluator.getFilePath(NetworkEvaluator.FILE_TYPE.IMAGE));

        testInfo.setText(networkEvaluator.getModelInfo().getScenarioName() + ": " +
                networkEvaluator.getModelInfo().getModelName() + "\ndata set: "
                + networkEvaluator.getModelInfo().getDataSetName() + "\n");
    }

    void initNetworkEvaluator() {
        ModelInfo modelInfo = new ModelInfo(scenarioName, modelName);
        networkEvaluator = new NetworkEvaluator(modelInfo);
        Map<String, PSNPEConfig> configs = PSNPEManager.getPSNPEConfig();
        if(!configs.containsKey(modelName)) {
            Util.displayAtToast(modelActivity.this, modelName + " not exist!!!", Toast.LENGTH_LONG);
            return;
        }
        networkEvaluator.setPsnpeConfig(configs.get(modelName));
        networkEvaluator.setEvaluationCallBacks(new EvaluationCallBacks() {
            @Override
            public void setExecuteStatus(final String status) {
                runOnUiThread(new Runnable() {
                    @Override
                    public void run() {
                        executeStatus.setText("Status: " + status);
                    }
                });
            }

            @Override
            public void showErrorText(final String error) {
                runOnUiThread(new Runnable() {
                    @Override
                    public void run() {
                        Util.displayAtToast(modelActivity.this, error, Toast.LENGTH_LONG);
                    }
                });

            }
        });
    }

    public void run(View v){
        // network config should be reset when run button click.

        // dynamic enable running mode view
        initRunningModeView();

        // execute network
        switch (networkEvaluator.getPsnpeConfig().transmissionMode) {
            case "sync":
                EvaluationSyncTask evaluationSyncTask = new EvaluationSyncTask(modelActivity.this);
                evaluationSyncTask.execute();
                break;
            case "inputOutputAsync":
                EvaluationInputOutputASyncTask evaluationInputOutputASyncTask = new EvaluationInputOutputASyncTask(modelActivity.this);
                evaluationInputOutputASyncTask.execute();
                break;
            case "outputAsync":
                EvaluationOutputASyncTask evaluationOutputASyncTask = new EvaluationOutputASyncTask(modelActivity.this);
                evaluationOutputASyncTask.execute();
                break;
            default:
                Log.e(TAG, "Mode can't be recognized");
                Util.displayAtToast(modelActivity.this,"Mode can't be recognized\n", Toast.LENGTH_LONG);
                break;
        }

    }

    public void stressRun(View v) {
        if(!networkEvaluator.getPsnpeConfig().transmissionMode.equals("sync")) {
            Util.displayAtToast(modelActivity.this, "Stress Test only support sync mode!", Toast.LENGTH_LONG);
            return;
        }

        stressTestRunning = !stressTestRunning;
        if(stressTestRunning) {
            initRunningModeView();
            EvaluationStressTask stressTask = new EvaluationStressTask(modelActivity.this);
            stressTask.execute();
            stressButton.setText("STOP");
        } else {
            networkEvaluator.stopStressTest();
            stressButton.setText("STRESS TEST");
        }
    }

    public void showResult(View v) {
        String res = networkEvaluator.getResult().toString();
        Intent intent = new Intent();
        intent.setClass(modelActivity.this, resultActivity.class);
        Bundle bundle = new Bundle();
        bundle.putString("result", networkEvaluator.getModelInfo().getScenarioName() + " " +
                networkEvaluator.getModelInfo().getModelName() + "\n\n" + res);
        intent.putExtra("key", bundle);
        startActivity(intent);
    }

    private void initRunningModeView() {
        //disable result button until run success
        runButton.setClickable(false);
        showResultButton.setClickable(false);
        showResultButton.setTextColor(Color.GRAY);
        executeStatus.setText("");
    }

    static class EvaluationSyncTask extends AsyncTask<Void, Void, Boolean> {
        private Boolean executeStatus = false;
        private WeakReference<modelActivity> activityWeakReference;

        EvaluationSyncTask(modelActivity context) {
            activityWeakReference = new WeakReference<>(context);
        }

        @Override
        protected void onPreExecute() {
            super.onPreExecute();
        }

        @Override
        protected Boolean doInBackground(Void... voids) {
            modelActivity activity = activityWeakReference.get();
            if(activity != null) {
                return executeStatus = activity.networkEvaluator.run();
            } else {
                return null;
            }
        }

        @Override
        protected void onPostExecute(Boolean aBoolean) {
            modelActivity activity = activityWeakReference.get();
            super.onPostExecute(aBoolean);
            if(activity != null) {
                //Display run result.
                if (executeStatus) {
                    Util.displayAtToast(activity, "Evaluation Success\n", Toast.LENGTH_LONG);
                    activity.showResultButton.setClickable(true);
                    activity.showResultButton.setTextColor(Color.BLACK);
                } else {
                    Util.displayAtToast(activity, "Evaluation failed", Toast.LENGTH_LONG);
                }
                activity.runButton.setClickable(true);
            }
        }
    }

    static class EvaluationOutputASyncTask extends AsyncTask<Void, Void, Boolean> {
        private String TAG = EvaluationOutputASyncTask.class.getSimpleName();
        private Boolean executeStatus = false;
        private WeakReference<modelActivity> activityWeakReference;

        EvaluationOutputASyncTask(modelActivity context) {
            activityWeakReference = new WeakReference<>(context);
        }

        @Override
        protected void onPreExecute() {
            super.onPreExecute();
            modelActivity activity = activityWeakReference.get();
            if (activity != null) {

            }
        }

        @Override
        protected Boolean doInBackground(Void... voids) {
            modelActivity activity = activityWeakReference.get();
            if (activity != null) {
                executeStatus = activity.networkEvaluator.runOutputAsync();
                if(executeStatus) {
                    try {
                        Thread.sleep(1000);
                    } catch (Exception e) {
                        e.printStackTrace();
                    }
                }
                return executeStatus;
            }
            return null;
        }

        @Override
        protected void onPostExecute(Boolean aBoolean) {
            super.onPostExecute(aBoolean);
            modelActivity activity = activityWeakReference.get();
            if (activity != null) {
                //Display run result.
                if (executeStatus) {
                    Util.displayAtToast(activity,"Evaluation Success\n", Toast.LENGTH_LONG);
                    activity.showResultButton.setClickable(true);
                    activity.showResultButton.setTextColor(Color.BLACK);
                } else {
                    Util.displayAtToast(activity,"Evaluation failed", Toast.LENGTH_LONG);
                }
                activity.runButton.setClickable(true);
            }
        }

        @Override
        protected void onProgressUpdate(Void... values) {
            super.onProgressUpdate(values);
            modelActivity activity = activityWeakReference.get();
            if (activity != null) {
                activity.runButton.setClickable(true);
            }
        }
    }

    static class EvaluationInputOutputASyncTask extends AsyncTask<Void, Void, Boolean> {
        private String TAG = EvaluationInputOutputASyncTask.class.getSimpleName();
        private Boolean executeStatus = false;
        private WeakReference<modelActivity> activityWeakReference;

        EvaluationInputOutputASyncTask(modelActivity context) {
            activityWeakReference = new WeakReference<>(context);
        }
        @Override
        protected void onPreExecute() {
            super.onPreExecute();
            modelActivity activity = activityWeakReference.get();
            if (activity != null) {
                Log.e(TAG, "null activity");
            }
        }

        @Override
        protected Boolean doInBackground(Void... voids) {
            modelActivity activity = activityWeakReference.get();
            if (activity != null) {
                executeStatus = activity.networkEvaluator.runInputOutputAsync();
                // handle free
                return executeStatus;
            }
            return null;
        }

        @Override
        protected void onPostExecute(Boolean aBoolean) {
            super.onPostExecute(aBoolean);
            modelActivity activity = activityWeakReference.get();
            if (activity != null) {
                //Display run result.
                if (executeStatus) {
                    Util.displayAtToast(activity,"Evaluation Success\n", Toast.LENGTH_LONG);
                    activity.showResultButton.setClickable(true);
                    activity.showResultButton.setTextColor(Color.BLACK);
                } else {
                    Util.displayAtToast(activity,"Evaluation failed", Toast.LENGTH_LONG);
                }
                activity.runButton.setClickable(true);
            }
        }
    }

    static class EvaluationStressTask extends AsyncTask<Void, Void, Boolean> {
        private Boolean executeStatus = false;
        private WeakReference<modelActivity> activityWeakReference;

        EvaluationStressTask(modelActivity context) {
            activityWeakReference = new WeakReference<>(context);
        }

        @Override
        protected void onPreExecute() {
            super.onPreExecute();
        }

        @Override
        protected Boolean doInBackground(Void... voids) {
            modelActivity activity = activityWeakReference.get();
            if(activity != null) {
                return executeStatus = activity.networkEvaluator.startStressTest();
            } else {
                return null;
            }
        }

        @Override
        protected void onPostExecute(Boolean aBoolean) {
            modelActivity activity = activityWeakReference.get();
            super.onPostExecute(aBoolean);
            if(activity != null) {
                //Display run result.
                if (executeStatus) {
                    Util.displayAtToast(activity, "Evaluation Success\n", Toast.LENGTH_LONG);
                } else {
                    Util.displayAtToast(activity, "Evaluation failed", Toast.LENGTH_LONG);
                    activity.stressButton.setText("STRESS TEST");
                    activity.stressTestRunning = !activity.stressTestRunning;
                }
                activity.runButton.setClickable(true);

            }
        }
    }
}
