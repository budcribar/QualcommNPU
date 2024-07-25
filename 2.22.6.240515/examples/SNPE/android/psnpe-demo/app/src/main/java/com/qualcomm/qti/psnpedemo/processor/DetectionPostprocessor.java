/*
 * Copyright (c) 2019-2021 Qualcomm Technologies, Inc.
 * All Rights Reserved.
 * Confidential and Proprietary - Qualcomm Technologies, Inc.
 */
package com.qualcomm.qti.psnpedemo.processor;

import android.util.Log;

import com.qualcomm.qti.psnpe.PSNPEManager;
import com.qualcomm.qti.psnpedemo.components.BenchmarkApplication;
import com.qualcomm.qti.psnpedemo.networkEvaluation.DetectionResult;
import com.qualcomm.qti.psnpedemo.networkEvaluation.EvaluationCallBacks;
import com.qualcomm.qti.psnpedemo.networkEvaluation.ModelInfo;
import com.qualcomm.qti.psnpedemo.networkEvaluation.Result;
import com.qualcomm.qti.psnpedemo.utils.BoundingBox;
import com.qualcomm.qti.psnpedemo.utils.ComputeUtil;
import com.qualcomm.qti.psnpedemo.utils.SimpleBox;
import com.qualcomm.qti.psnpedemo.utils.Util;

import org.json.JSONArray;
import org.json.JSONException;
import org.json.JSONObject;

import java.io.File;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class DetectionPostprocessor extends PostProcessor {
    private static String TAG = "DetectionPostprocessor";
    private String modelName;
    private String groundTruthPath;
    private String imageLabelPath;

    private String dataSetName;
    private double map;
    public float confidenceThreshold = (float)0.0;
    public boolean isCrop = false;
    private static double IOU = 0.5;
    List<Map<String, float []>> results = new ArrayList<>();
    //Integer: class BoundingBox:confidence & bbox & image
    public static Map<Integer, ArrayList<BoundingBox>> predictions;
    //Integer: class String:imageId List:bbox
    private static Map<Integer, HashMap<String, List<List>>> groundTruths;

    List<String> ImageIds = new ArrayList<>();

    public DetectionPostprocessor(EvaluationCallBacks evaluationCallBacks, ModelInfo modelInfo, int imageNumber) {
        super(imageNumber);
        this.evaluationCallBacks = evaluationCallBacks;
        modelName= modelInfo.getModelName();
        dataSetName = modelInfo.getDataSetName();
        if(dataSetName == "voc"){
            String packagePath = BenchmarkApplication.getCustomApplicationContext().getExternalFilesDir("").getAbsolutePath();
            groundTruthPath = packagePath + "/datasets/" + dataSetName + "/voc_annotations.json";
            imageLabelPath = packagePath + "/datasets/" + dataSetName + "/voc_images.json";
        }
        else if(dataSetName == "coco"){
            String packagePath = BenchmarkApplication.getCustomApplicationContext().getExternalFilesDir("").getAbsolutePath();
            groundTruthPath = packagePath + "/datasets/" + dataSetName + "/coco_annotations.json";
            imageLabelPath = packagePath + "/datasets/" + dataSetName + "/coco_images.json";
        }

    }
    @Override
    public boolean postProcessResult(ArrayList<File> inputImages) {
        Log.d(TAG, "start into detection post process!");
        int imageNum = inputImages.size();
        List<Map<String,float []>> outputs = new ArrayList<>(imageNum);
        Map<String, float []> outputMap = null;
        for(int i=0; i< imageNum; i++) {
            /* output:
             * <outputBuffer1><outputBuffer2>...<imageBulkSize/batchSize>
             * split output and handle one by one.
             */
            outputs.add(readOutput(i));
        }

        List<String> Id=new ArrayList<>();
        for(int j = 0;j<inputImages.size();j++){
            String imageName=inputImages.get(j).getName();
            String imageId=imageName.split("\\.")[0];
            Id.add(j,imageId);
        }
        map = EvaluateAccuracy(outputs,Id);
        return true;
    }

    @Override
    public void setResult(Result result) {
        DetectionResult dresult= (DetectionResult)result;
        dresult.setMap(map);
    }

    @Override
    public void resetResult(){}

    @Override
    public void getOutputCallback(String fileName, Map<String, float[]> outputs) {

    }

    public Map<Integer, HashMap<String, List<List>>> parseGroundTruth(String gtPath, Map<String, String> imagesToId) throws JSONException {
        if(gtPath.equals("") ||imagesToId == null) {
            Log.e(TAG, "gtPath or imagesToId ids null");
            return null;
        }
        Map<Integer, HashMap<String, List<List>>> groundTruthData = new HashMap<>();
        String gtData = Util.getDatafromFile(gtPath);
        JSONArray jsonArray;
        JSONObject jsonObject;
        if(0 != gtData.length()) {
            jsonObject = new JSONObject(gtData);
            jsonArray = jsonObject.getJSONArray("annotations");

        }
        else {
            Log.e(TAG,"could not find the ground truth file: " + gtPath);
            return null;
        }


        String imageId;
        int category_id;
        for(int i=0;i<jsonArray.length();i++) {
            JSONObject item = (JSONObject) jsonArray.getJSONObject(i);
            JSONArray bbox_jsonObject = item.getJSONArray("bbox");
            List<Float> bbox = new ArrayList<Float>();
            if(bbox_jsonObject != null) {
                bbox.add((float)bbox_jsonObject.getDouble(0));
                bbox.add((float)bbox_jsonObject.getDouble(1));
                bbox.add((float)(bbox_jsonObject.getDouble(0)+bbox_jsonObject.getDouble(2)));
                bbox.add((float)(bbox_jsonObject.getDouble(1)+bbox_jsonObject.getDouble(3)));
            }
            imageId = item.getString("image_id");
            category_id = item.getInt("category_id");
            if(imagesToId.containsKey(imageId)) {
                if(!groundTruthData.containsKey(category_id)) {
                    HashMap<String, List<List>> tmpMap;
                    tmpMap = new HashMap<String, List<List>>();
                    groundTruthData.put(category_id, tmpMap);
                }
                if(!groundTruthData.get(category_id).containsKey(imageId)) {
                    List<List> tmpList;
                    tmpList = new ArrayList<List>();
                    groundTruthData.get(category_id).put(imageId, tmpList);
                }
                List<List> boxes = groundTruthData.get(category_id).get(imageId);
                boxes.add(bbox);
            }

        }
        return groundTruthData;
    }

    public Map<Integer, ArrayList<BoundingBox>> getPredictionResult(List<Map<String, float[]>> outputs, List<String> imageIds) throws JSONException {
        String imagesFile=Util.getDatafromFile(imageLabelPath);
        JSONObject json_object= new JSONObject(imagesFile);
        Map<Integer, ArrayList<BoundingBox>> predictionData = new HashMap<Integer, ArrayList<BoundingBox>>();
        if(results == null || imageIds == null) {
            Log.d(TAG, "results or images ids null");
            return null;
        }

        for(int i = 0; i < outputs.size(); i++) {
            Map<String, float []> output = outputs.get(i);

            float[]  bboxArray = null;
            float[]  scoreArray = null;
            float[]  classArrayTmp = null;
            if(modelName.contains("yolo")){
                float[]  feature_map_52 = output.get("yolov3/yolov3_head/Conv_22/Conv2D:0");
                float[]  feature_map_26 = output.get("yolov3/yolov3_head/Conv_14/Conv2D:0");
                float[]  feature_map_13 = output.get("yolov3/yolov3_head/Conv_6/Conv2D:0");
                /* TODO: add yolo post process */
                bboxArray =new float[1*4];
                scoreArray =new float[1];
                classArrayTmp=new float[1];

            }
            else if(modelName.contains("ssd")){
                for(String key: output.keySet()) {
                    if(key.contains("boxes") && bboxArray == null){
                        bboxArray =output.get(key);
                    }
                    else if(key.contains("scores") && scoreArray == null){
                        scoreArray = output.get(key);
                    }
                    else if(key.contains("classes") && classArrayTmp == null){
                        classArrayTmp =output.get(key);
                    }
                }
                if(bboxArray == null || scoreArray == null || classArrayTmp == null){
                    Log.e(TAG,"can't find all outputs layer");
                    return null;
                }

            } else {
                Log.e(TAG,"model "+modelName+" is not supported");
                return null;
            }

            int[] classArray = ComputeUtil.mathFloor(classArrayTmp);
            if(dataSetName =="coco"){
                for(int k = 0; k < classArray.length; k++) {
                    classArray[k] = classArray[k] + 1;
                }
            }
            ArrayList boundingBoxList;
            for(int j = 0; j < scoreArray.length; j++) {
                if(scoreArray[j] == 0) {
                    break;
                }
                float topConfidence = scoreArray[j];
                if(topConfidence < confidenceThreshold) {
                    continue;
                }
                float topXmin, topYmin, topW, topH;
                topXmin = (float)0.0;
                topYmin = (float)0.0;
                topW = (float)0.0;
                topH = (float)0.0;
                String id = "";
                if(!isCrop) {
                    id = imageIds.get(i);
                    id = id.replaceAll("^(0+)", "");
                    JSONObject tmp = null;
                    tmp = json_object.getJSONObject(id);
                    int Height = 0, Width = 0;
                    if(tmp != null) {
                        Height = json_object.getJSONObject(id).getInt("Height");
                        Width = json_object.getJSONObject(id).getInt("Width");
                    }
                    else {
                        Height = 0;
                        Width = 0;
                        Log.e(TAG, "image not found in annotations, image id is:" + id);
                    }
                    topXmin = (bboxArray[(j * 4) + 1]) * Width;
                    topYmin =  (bboxArray[(j * 4)]) * Height;
                    topW = (bboxArray[(j * 4) + 3] - bboxArray[(j * 4) + 1]) * Width;
                    topH = (bboxArray[(j * 4) + 2] - bboxArray[(j * 4)]) * Height;
                }
                Integer topCategory;
                topCategory = Float.valueOf(classArray[j]).intValue();

                ArrayList<Float> box = new ArrayList<>();
                box.add(topXmin);
                box.add(topYmin);
                box.add((topXmin + topW));
                box.add((topYmin + topH));
                BoundingBox boundingBox = new BoundingBox(topConfidence, id, box);
                if(!predictionData.containsKey(topCategory)) {
                    predictionData.put(topCategory, new ArrayList<BoundingBox>());
                }
                boundingBoxList = predictionData.get(topCategory);
                boundingBoxList.add(boundingBox);
                predictionData.put(topCategory, boundingBoxList);
            }
        }
        return predictionData;

    }
    public double EvaluateAccuracy(List<Map<String, float[]>> outputs,List<String> ImageId) {
        // Don't directly modify groundTruth/labels value, it maybe override multiple time.
        ImageIds.clear();
        double map = 0.0;
        Map<String, String> imagesToId = new HashMap<>();
        for(int i = 0; i < outputs.size(); i++) {
            String id= ImageId.get(i).replaceAll("^(0+)", "");
            ImageIds.add(id);
            imagesToId.put(id, id);
        }
        try {
            groundTruths = parseGroundTruth(groundTruthPath, imagesToId);
            predictions = getPredictionResult(outputs, ImageId);
        } catch (JSONException e) {
            e.printStackTrace();
        }
        double ap;
        for(Integer category: predictions.keySet()) {
            if(!groundTruths.containsKey(category)){
                continue;
            }
            ap = getAveragePrecision(category, groundTruths, predictions, IOU);
            Log.d(TAG, String.valueOf(category)+" "+String.valueOf(ap));
            map += ap;
        }
        map = map/predictions.size();
        return map;
    }

    public  double getAveragePrecision(Integer category, Map<Integer, HashMap<String, List<List>>> groundtruth, Map<Integer, ArrayList<BoundingBox>> predictions, double minOverlap) {
        double ap = 0.0;
        int gtCounter = 0;
        HashMap<String, List<List>> groundTruthDict = groundtruth.get(category);
        ArrayList<BoundingBox> predictionDict = predictions.get(category);
        Map<String, List<SimpleBox>> groundTruthData = new HashMap<String, List<SimpleBox>>();
        for(String imageId: groundTruthDict.keySet()) {
            List<SimpleBox> tmpFile = new ArrayList<SimpleBox>();
            for(List<Float> box: groundTruthDict.get(imageId)) {
                SimpleBox sb = new SimpleBox(box, false);
                tmpFile.add(sb);
                gtCounter++;
            }
            groundTruthData.put(imageId, tmpFile);
        }
        Collections.sort(predictionDict, new Comparator<BoundingBox>() {
            public int compare(BoundingBox o1, BoundingBox o2) {
                if(o1.getConfidence()>o2.getConfidence()){
                    return -1;
                }else if(o1.getConfidence()<o2.getConfidence()){
                    return 1;
                }else{
                    return 0;
                }
            }
        });
        int lenPredictionsDict = predictionDict.size();
        int[] TP = new int[lenPredictionsDict];
        for(int i=0;i<TP.length;i++){
            TP[i]=0;
        }
        int[] FP = new int[lenPredictionsDict];
        for(int i=0;i<FP.length;i++){
            FP[i]=0;
        }
        int TruePositiveCounter = 0;
        String imageId;
        ArrayList<Float> bb;
        ArrayList<Float> bbGT = null;

        ArrayList<SimpleBox> grouthTruthInImage;
        for(int i = 0; i < lenPredictionsDict; i++) {
            imageId = predictionDict.get(i).getImageId();
            bb = predictionDict.get(i).getBox();
            Float ovMax = (float)-1.0;
            SimpleBox gtMatch = null;
            Float iw, ih, ua, ov;
            if(groundTruthData.containsKey(imageId)) {
                grouthTruthInImage = (ArrayList<SimpleBox>) groundTruthData.get(imageId);
                for(int j = 0; j < grouthTruthInImage.size(); j++) {
                    bbGT = (ArrayList<Float>) grouthTruthInImage.get(j).getBox();
                    List<Float> bi = new ArrayList<Float>();
                    bi.add(Math.max(bb.get(0),bbGT.get(0)));
                    bi.add(Math.max(bb.get(1),bbGT.get(1)));
                    bi.add(Math.min(bb.get(2),bbGT.get(2)));
                    bi.add(Math.min(bb.get(3),bbGT.get(3)));
                    iw = bi.get(2) - bi.get(0) + 1;
                    ih = bi.get(3) - bi.get(1) + 1;
                    if(iw > 0 && ih > 0) {
                        ua = (bb.get(2) - bb.get(0) + 1) * (bb.get(3) - bb.get(1) + 1)
                                + (bbGT.get(2) - bbGT.get(0) + 1) * (bbGT.get(3) - bbGT.get(1) + 1)
                                - iw * ih;
                        ov = iw * ih / ua;
                        if(ov > ovMax) {
                            ovMax = ov;
                            gtMatch = grouthTruthInImage.get(j);
                        }
                    }

                }

            }
            if(ovMax >= minOverlap) {
                if(!gtMatch.isUsed()) {
                    TP[i] = 1;
                    gtMatch.setUsed(true);
                    TruePositiveCounter += 1;
                }
                else {
                    FP[i] = 1;
                }
            }
            else {
                FP[i]  = 1;
            }



        }
        int cumSum = 0;
        float eps = (float)2.220446049250313e-16;
        for(int i = 0; i < FP.length; i++) {
            int val = FP[i];
            FP[i] += cumSum;
            cumSum += val;
        }
        cumSum = 0;
        for(int i = 0; i < TP.length; i++) {
            int val = TP[i];
            TP[i] += cumSum;
            cumSum += val;
        }
        float[] rec = new float[TP.length];
        for(int i = 0; i < TP.length; i++) {
            rec[i] = (float)TP[i]/gtCounter;
        }
        float[] prec = new float[TP.length];
        for(int i = 0; i < TP.length; i++) {
            prec[i] = (float)TP[i]/(FP[i] + TP[i] + eps);
        }
        if(dataSetName=="voc"){
            ap = getVocAp(rec, prec);
        } else{
            ap = getCocoAp(rec, prec);
        }

        return ap;
    }
    public static float getVocAp(float[] rec, float[] prec) {
        float ap = (float)0.0;
        float[] mrec = new float[rec.length + 2];
        float[] mprec = new float[prec.length + 2];
        mrec[0] = 0;
        for(int i = 0; i < rec.length; i++) {
            mrec[i+1] = rec[i];
        }
        mrec[mrec.length -1] = 1;
        mprec[0] = 0;
        for(int i = 0; i < prec.length; i++) {
            mprec[i+1] = prec[i];
        }
        mprec[mprec.length -1] = 0;
        for(int i = mprec.length - 2; i >=0; i--) {
            mprec[i] = Math.max(mprec[i], mprec[i+1]);
        }
        List<Integer> iList = new ArrayList<Integer>();
        for(int i = 1; i < mrec.length; i++) {
            if(mrec[i] != mrec[i-1]) {
                iList.add(i);
            }
        }
        for(int i: iList) {
            ap += (float)(mrec[i]-mrec[i-1])*mprec[i];
        }
        return ap;
    }

    public static float getCocoAp(float[] rec, float[] prec) {
        float ap = (float)0.0;
        double[] recThrs = new double[101];
        for(int i = 0; i < recThrs.length; i++) {
            recThrs[i] = (1.0 * i)/100;
        }
        int nd = prec.length;
        double[] q = new double[recThrs.length];
        for(int i = 0; i < q.length; i++) {
            q[i] = 0;
        }
        double[] precTmp = new double[prec.length + 1];
        for(int i = 0; i < prec.length; i++) {
            precTmp[i] = prec[i];
        }
        for(int i = nd - 1; i >0; i--) {
            if(precTmp[i] > precTmp[i - 1]) {
                precTmp[i - 1] = precTmp[i];
            }
        }
        precTmp[precTmp.length - 1] = prec[prec.length - 1];
        int[] inds = new int[recThrs.length];
        for(int i = 0; i < inds.length; i++) {
            inds[i] = ComputeUtil.searchSorted(rec, recThrs[i]);
        }
        int count = 0;
        for(int i:inds) {
            try {
                q[count++] = precTmp[i];
            } catch (Exception e){
                e.printStackTrace();
            }
        }
        ap = (float) ComputeUtil.getAverage(q);
        return ap;
    }
}