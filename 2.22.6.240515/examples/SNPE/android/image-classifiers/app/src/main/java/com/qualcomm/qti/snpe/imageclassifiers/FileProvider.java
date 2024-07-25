/*
 * Copyright (c) 2023 Qualcomm Technologies, Inc.
 * All Rights Reserved.
 * Confidential and Proprietary - Qualcomm Technologies, Inc.
 */
package com.qualcomm.qti.snpe.imageclassifiers;

import android.net.Uri;
import android.database.Cursor;
import android.content.ContentValues;
import android.content.ContentProvider;
import android.support.annotation.Nullable;
import android.support.annotation.NonNull;

public class FileProvider extends ContentProvider {

    public static final String CONTENT_PROVIDER_AUTHORITY = "com.qualcomm.qti.snpe.imageclassifiers.FileProvider";

    @Override
    public boolean onCreate() {
        return true;
    }

    @Nullable
    @Override
    public Cursor query(@NonNull Uri uri, @Nullable String[] projection, @Nullable String selection, @Nullable String[] selectionArgs, @Nullable String sortOrder) {
        return null;
    }

    @Nullable
    @Override
    public String getType(@NonNull Uri uri) {
        return null;
    }

    @Nullable
    @Override
    public Uri insert(@NonNull Uri uri, @Nullable ContentValues values) {
        return null;
    }

    @Override
    public int delete(@NonNull Uri uri, @Nullable String selection, @Nullable String[] selectionArgs) {
        return 0;
    }

    @Override
    public int update(@NonNull Uri uri, @Nullable ContentValues values, @Nullable String selection, @Nullable String[] selectionArgs) {
        return 0;
    }

    public static Uri getFileFromUri(Uri uri) {
        Uri.Builder builder = new Uri.Builder()
                .authority(CONTENT_PROVIDER_AUTHORITY)
                .scheme("file")
                .path(uri.getPath())
                .fragment(uri.getFragment());

        return builder.build();
    }
}