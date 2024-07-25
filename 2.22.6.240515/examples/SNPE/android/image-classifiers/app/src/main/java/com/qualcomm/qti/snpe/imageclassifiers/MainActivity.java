/*
 * Copyright (c) 2021 Qualcomm Technologies, Inc.
 * All Rights Reserved.
 * Confidential and Proprietary - Qualcomm Technologies, Inc.
 */
package com.qualcomm.qti.snpe.imageclassifiers;

import android.app.Application;
import com.qualcomm.qti.snpe.SNPE;
import com.qualcomm.qti.snpe.NeuralNetwork;

import android.app.Activity;
import android.app.FragmentTransaction;
import android.os.Bundle;

public class MainActivity extends Activity {

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        // The following code demonstrates the usage of Java logging APIs to initialize logging to verbose
        // level and then set it to info level. Logging is terminated when the app activity is destroyed.
        SNPE.logger.initializeLogging((Application) getApplicationContext(), NeuralNetwork.LogLevel.LOG_VERBOSE);
        SNPE.logger.setLogLevel(NeuralNetwork.LogLevel.LOG_INFO);

        if (savedInstanceState == null) {
            final FragmentTransaction transaction = getFragmentManager().beginTransaction();
            transaction.add(R.id.main_content, ModelCatalogueFragment.create());
            transaction.commit();
        }
    }

    public void displayModelOverview(final Model model, boolean unsignedPD) {
        final FragmentTransaction transaction = getFragmentManager().beginTransaction();
        transaction.replace(R.id.main_content, ModelOverviewFragment.create(model, unsignedPD));
        transaction.addToBackStack(null);
        transaction.commit();
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        SNPE.logger.terminateLogging();
    }
}
