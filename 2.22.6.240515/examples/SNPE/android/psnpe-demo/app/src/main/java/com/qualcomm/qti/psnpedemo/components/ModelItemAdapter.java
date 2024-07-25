/*
 * Copyright (c) 2019 Qualcomm Technologies, Inc.
 * All Rights Reserved.
 * Confidential and Proprietary - Qualcomm Technologies, Inc.
 */
package com.qualcomm.qti.psnpedemo.components;

import android.content.Context;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.ArrayAdapter;
import android.widget.ImageView;
import android.widget.TextView;

import com.qualcomm.qti.psnpedemo.R;

import java.util.List;

public class ModelItemAdapter extends ArrayAdapter<ModelListItem> {
    private int newResourceId;
    public ModelItemAdapter(Context context, int resourceId, List<ModelListItem> modelList){
        super(context, resourceId, modelList);
        newResourceId = resourceId;
    }

    @Override
    public View getView(int position, View convertView, ViewGroup parent) {
        ModelListItem modelItem = getItem(position);
        View view = LayoutInflater.from(getContext()).inflate(newResourceId, parent, false);

        TextView modelName = view.findViewById(R.id.modelName);

        modelName.setText(modelItem.getScenarioName() + ": " + modelItem.getModelName());
        return view;
    }
}
