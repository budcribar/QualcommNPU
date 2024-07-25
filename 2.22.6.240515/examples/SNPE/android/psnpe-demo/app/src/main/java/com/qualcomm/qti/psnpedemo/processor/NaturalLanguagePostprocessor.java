/*
 * Copyright (c) 2022 Qualcomm Technologies, Inc.
 * All Rights Reserved.
 * Confidential and Proprietary - Qualcomm Technologies, Inc.
 */
package com.qualcomm.qti.psnpedemo.processor;

import android.util.Log;
import com.qualcomm.qti.psnpe.PSNPEManager;
import com.qualcomm.qti.psnpedemo.components.BenchmarkApplication;
import com.qualcomm.qti.psnpedemo.networkEvaluation.EvaluationCallBacks;
import com.qualcomm.qti.psnpedemo.networkEvaluation.ModelInfo;
import com.qualcomm.qti.psnpedemo.networkEvaluation.Result;


import java.io.IOException;
import java.io.Reader;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Collections;
import java.util.Locale;

import java.io.File;
import java.io.FileInputStream;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.util.Objects;
import java.util.stream.Collectors;
import java.util.stream.Stream;

import com.qualcomm.qti.psnpedemo.networkEvaluation.NaturalLanguageResult;
import com.qualcomm.qti.psnpedemo.utils.Answer;
import com.qualcomm.qti.psnpedemo.utils.Sample;

import org.json.JSONArray;
import org.json.JSONException;
import org.json.JSONObject;


public class NaturalLanguagePostprocessor extends PostProcessor{
    private final String groundTruthPath;
    private float F1;
    private float ExactMatch;
    private static final String TAG = "NaturalLanguagePostprocessor";
    private static final int PREDICT_ANS_NUM = 20;
    private static final int MAX_ANS_LEN = 30;
    private static final int MAX_SEQ_LEN = 384;

    public NaturalLanguagePostprocessor(EvaluationCallBacks evaluationCallBacks, ModelInfo modelInfo, int imageNumber)  {
        super(imageNumber);
        String modelName = modelInfo.getModelName();
        String dataSetName = modelInfo.getDataSetName();
        String packagePath = BenchmarkApplication.getCustomApplicationContext().getExternalFilesDir("").getAbsolutePath();
        groundTruthPath = packagePath + "/datasets/" + dataSetName + "/labels.txt";
        this.evaluationCallBacks = evaluationCallBacks;
    }

    @Override
    public boolean postProcessResult(ArrayList<File> inputImages) {
        //The eval_mini,groundtruth_mini file contains only 75 data. cover the dataset we use for test
        String evalPath = "/storage/emulated/0/Android/data/com.qualcomm.qti.psnpedemo/files/datasets/squad/eval_mini.json";
        String gtPath = "/storage/emulated/0/Android/data/com.qualcomm.qti.psnpedemo/files/datasets/squad/groundtruth_mini.json";
        List<List<String>> groundTruths = getGrounthTruth(gtPath);
        List<Sample> samples = getSamples(evalPath);
        float finalScore = 0.0f;
        float exactMatch = 0.0f;
        int imageNum = inputImages.size();
        Log.i(TAG, "postProcessResult doimage number: " + imageNum);
        String[] outputNames = PSNPEManager.getOutputTensorNames();
        //for each data
        for(int i = 0; i < imageNum; i++) {
            float[] output = readOutput(i).get(outputNames[0]);
            float[] unstack0  = new float[384];
            float[] unstack1 = new float[384];
            System.arraycopy(output,0,unstack0,0,384);
            System.arraycopy(output,384,unstack1,0,384);
            float[] startLogits = unstack0;
            float[] endLogits = unstack1;
            ArrayList<Answer> answers = getBestAnswers(startLogits, endLogits, samples.get(i));
            if(answers.size() >0) {
                finalScore += F1Score(groundTruths.get(i), answers.get(0).text);
                exactMatch += ExactMatchScore(groundTruths.get(i), answers.get(0).text);
            }
        }

        F1 = finalScore /imageNum;
        ExactMatch = exactMatch/imageNum;
        return true;
    }

    @Override
    public void setResult(Result result) {
        NaturalLanguageResult cresult = (NaturalLanguageResult)result;
        cresult.setF1((float)F1);
        cresult.setExactMatch((float)ExactMatch);
        Log.d(TAG, "F1: " + F1 + " ExactMatch: " +ExactMatch  );
    }

    @Override
    public void resetResult(){}

    @Override
    public void getOutputCallback(String fileName, Map<String, float[]> outputs) {
        Log.i(TAG, "Async output postprocessor finished");
    }


    private static String convertBack(Sample sample, int start, int end) {
        int shiftedStart = start + 1;
        int shiftedEnd = end + 1;
        List<String> tokTokens = sample.tokTokens.subList(start, end + 1);
        String tokText  = String.join(" ", tokTokens);
        //startIndex is the orig_doc_start
        //endIndex is the orig_doc_end
        int startIndex = sample.tokenToOrigMap.get(shiftedStart);
        int endIndex = sample.tokenToOrigMap.get(shiftedEnd);

        List<String> origTokens = sample.origTokens.subList(startIndex, endIndex + 1);
        String origText = String.join(" ", origTokens);

        return getFinaltext(tokText, origText);
    }

    private static String getFinaltext(String predText, String origText) {
        String tokText = origText;
        int startPosition = tokText.indexOf(predText);
        if (startPosition == -1){
            return origText;
        }
        int endPosition = startPosition + predText.length() - 1;
        HashMap<Integer, Integer> origNsToSMap = stripSpaces(origText);
        HashMap<Integer, Integer> tokNsToSMap = stripSpaces(tokText);
        HashMap<Integer, Integer> tokSToNsMap = new HashMap<>();
        for (Integer key :tokNsToSMap.keySet()){
            tokSToNsMap.put(tokNsToSMap.get(key),key);
        }
        Integer origStartPosition = -1;
        if(tokSToNsMap.containsKey(startPosition)){
            Integer nsStartPosition = tokSToNsMap.get(startPosition);
            if (origNsToSMap.containsKey(nsStartPosition)){
                origStartPosition = origNsToSMap.get(nsStartPosition);
            }
        }
        if(origStartPosition == -1){
            return origText;
        }

        Integer origEndPosition = -1;
        if(tokSToNsMap.containsKey(endPosition)){
            Integer nsEndPosition = tokSToNsMap.get(endPosition);
            if (origNsToSMap.containsKey(nsEndPosition)){
                origEndPosition = origNsToSMap.get(nsEndPosition);
            }
        }
        if(origEndPosition == -1){
            return origText;
        }
        String outputText = origText.substring(origStartPosition, origEndPosition + 1);

        return outputText;
    }
    private static HashMap<Integer, Integer> stripSpaces(String text){
        ArrayList<Object> nsChars = new ArrayList<>();
        HashMap<Integer,Integer> nsToSMap = new HashMap<>() ;
        for(int i =0; i <text.length();i++){
            if(text.substring(i,i+1).contains(" ")){
                continue;
            }
            nsToSMap.put(nsChars.size(),i);
            nsChars.add(text.substring(i,i+1));
        }
        return nsToSMap;
    }


    private int[] getBestIndex(float[] logits) {
        List<Answer.Pos> tmpList = new ArrayList<Answer.Pos>();
        for (int i = 0; i < MAX_SEQ_LEN; i++) {
            tmpList.add(new Answer.Pos(i, i, logits[i]));
        }
        Collections.sort(tmpList);

        int[] indexes = new int[PREDICT_ANS_NUM];
        for (int i = 0; i < PREDICT_ANS_NUM; i++) {
            indexes[i] = tmpList.get(i).start;
        }

        return indexes;
    }

    private ArrayList<Answer> getBestAnswers(
            float[] startLogits, float[] endLogits, Sample sample) {
        // Model uses the closed interval [start, end] for indices.
        int[] startIndexes = getBestIndex(startLogits);
        int[] endIndexes = getBestIndex(endLogits);
        ArrayList<Answer.Pos> origResults = new ArrayList<>();
        for (int start : startIndexes) {
            for (int end : endIndexes) {
                if (!sample.tokenToOrigMap.containsKey(start + 1)) {
                    continue;
                }
                if (!sample.tokenToOrigMap.containsKey(end + 1)) {
                    continue;
                }
                if (end < start) {
                    continue;
                }
                int length = end - start + 1;
                if (length > MAX_ANS_LEN) {
                    continue;
                }
                origResults.add(new Answer.Pos(start, end, startLogits[start] + endLogits[end]));
            }
        }

        Collections.sort(origResults);

        ArrayList<Answer> answers = new ArrayList<>();
        for (int i = 0; i < origResults.size(); i++) {
            if (i >= PREDICT_ANS_NUM) {
                break;
            }
            String convertedText;
            //sample is the feature of each data
            if (origResults.get(i).start > 0) {
                convertedText = convertBack(sample, origResults.get(i).start, origResults.get(i).end);
            } else {
                convertedText = "";
            }
            Answer ans = new Answer(convertedText, origResults.get(i));
            answers.add(ans);
        }
        return answers;
    }


    public static List<String> getDocTokens(List<String> inputTokens, List<Integer> sequence) {
        String mergeToken = "";
        List<String> result = new ArrayList<String>();
        for(int k = 0; k < sequence.size() - 1; k++) {
            mergeToken += inputTokens.get(k);
            if(sequence.get(k) != sequence.get(k+1)) {
                result.add(mergeToken);
                mergeToken = "";
            }
        }
        if(mergeToken == "") {
            result.add(inputTokens.get(inputTokens.size()-1));
        }
        else
        {
            mergeToken += inputTokens.get(inputTokens.size()-1);
            result.add(mergeToken);
        }
        for(int i = 0; i < result.size(); i++) {
            result.set(i, result.get(i).replace("##","").replace(" ##",""));
        }
        return result;
    }
    public static String readAssetsFileAsString(String runtimeConfigFilePath) throws IOException {
        InputStream inputStream = new FileInputStream(runtimeConfigFilePath);
        Reader reader = new InputStreamReader(inputStream);
        StringBuilder sb = new StringBuilder();
        char[] buffer = new char[1638400];

        int len;
        while((len = reader.read(buffer)) > 0) {
            sb.append(buffer, 0, len);
        }
        reader.close();
        return sb.toString();
    }

    List<Sample> getSamples(String evalPath) {
        List<Sample> samples = new ArrayList<Sample>();
        JSONArray jsonArray;
        List<Integer> inputIds = new ArrayList<>();
        List<Integer> inputMask= new ArrayList<>();
        List<Integer> segmentIds = new ArrayList<>();
        List<String> origTokens = new ArrayList<>();
        List<String> qaTokens = new ArrayList<>();
        List<Integer> tokenToorigMapKeys = new ArrayList<>();
        List<Integer> tokenToorigMapValues = new ArrayList<>();
        try{
            String evalData = readAssetsFileAsString(evalPath);
            JSONObject qaSamples = new JSONObject(evalData);
            if(0 != evalData.length()) {

                jsonArray = qaSamples.getJSONArray("eval_data");
                if (jsonArray != null) {
                    for (int i = 0; i < jsonArray.length(); i++) {

                        JSONObject qaSample = jsonArray.getJSONObject(i);
                        JSONArray inputIdsTmp = qaSample.getJSONArray("input_ids");
                        JSONArray inputMaskTmp = qaSample.getJSONArray("input_mask");
                        JSONArray segmentIdsTmp = qaSample.getJSONArray("segment_ids");
                        JSONArray tokenToorigMapKeysTmp = qaSample.getJSONArray("token_to_orig_map_keys");
                        JSONArray tokenToorigMapValuesTmp = qaSample.getJSONArray("token_to_orig_map_values");
                        //Use the for loop to build the inputIds and other lists
                        for (int k = 0; k< inputIdsTmp.length(); k++){
                            inputIds.add(Integer.valueOf(inputIdsTmp.get(k).toString()));
                            inputMask.add(Integer.valueOf(inputMaskTmp.get(k).toString()));
                            segmentIds.add(Integer.valueOf(segmentIdsTmp.get(k).toString()));
                        }
                        for (int j = 0;j< tokenToorigMapKeysTmp.length();j++){
                            tokenToorigMapKeys.add(Integer.valueOf(tokenToorigMapKeysTmp.get(j).toString()));
                        }
                        for (int k = 0;k< tokenToorigMapValuesTmp.length();k++){
                            tokenToorigMapValues.add(Integer.valueOf(tokenToorigMapValuesTmp.get(k).toString()));
                        }
                        JSONArray qaTokens_ = qaSample.getJSONArray("tokens");
                        for (int k = 0;k< qaTokens_.length();k++){
                            qaTokens.add(qaTokens_.get(k).toString());
                        }
                        origTokens = qaTokens.subList(qaTokens.indexOf("[SEP]") + 1, qaTokens.size() - 1);

                        List<String> docTokens = getDocTokens(origTokens, tokenToorigMapValues);
                        Map<Integer, Integer> tokenToorigMap = new HashMap<Integer, Integer>();
                        for (int j = 0; j < tokenToorigMapKeys.size(); j++) {
                            tokenToorigMap.put(tokenToorigMapKeys.get(j) + 1, tokenToorigMapValues.get(j));
                        }
                        Sample sample = new Sample(inputIds, inputMask, segmentIds,
                                docTokens, qaTokens ,tokenToorigMap);
                        samples.add(sample);
                        //Clear the inputIds and other data, and wait for the next sample to enter
                        inputIds = new ArrayList<>();
                        inputMask= new ArrayList<>();
                        segmentIds = new ArrayList<>();
                        origTokens = new ArrayList<>();
                        qaTokens = new ArrayList<>();
                        tokenToorigMapKeys = new ArrayList<>();
                        tokenToorigMapValues = new ArrayList<>();
                    }
                }
            }
        } catch (JSONException | IOException e) {
            e.printStackTrace();
        }


        return samples;
    }

    List<List<String>> getGrounthTruth(String groundPath) {
        List<List<String>> groundTruths = new ArrayList<List<String>>();
        try{
            String groundTruthData = null;
            groundTruthData = readAssetsFileAsString(groundPath);
            JSONArray jsonArray;
            List<String> groundTruthAns;
            JSONObject grounTruthItem =new JSONObject(groundTruthData); ;
            if(0 != groundTruthData.length()) {
                jsonArray = grounTruthItem.getJSONArray("groudtruth");
                if (jsonArray != null) {
                    for (int i = 0; i < jsonArray.length(); i++) {
                        grounTruthItem = jsonArray.getJSONObject(i);
                        JSONArray groundTruthAns2 = grounTruthItem.getJSONArray("answers");
                        String jsonString =  groundTruthAns2.toString();//JSONArray->String
                        groundTruthAns = Stream.of(jsonString).collect(Collectors.toList());//String->List<String>
                        groundTruths.add(groundTruthAns);
                    }
                }
            }
        }catch(IOException | JSONException e){
            e.printStackTrace();
        }

        return groundTruths;
    }

    String Normalize(String text) {
        String removePuncText = text.replaceAll("\\p{Punct}", "");
        String[] removePuncTextSplits = removePuncText.split(" ");
        String removeArticle = "";
        List<String> removeArticles= new ArrayList<String>();
        for(String removePuncTextSplit: removePuncTextSplits) {
            if(removePuncTextSplit.equals("a") || removePuncTextSplit.equals("an") || removePuncTextSplit.equals("the")
                    || removePuncTextSplit.equals("A") || removePuncTextSplit.equals("An") || removePuncTextSplit.equals("The"))  {
                continue;
            }
            removeArticles.add(removePuncTextSplit);
        }

        String[] whiteSpaceFixs = String.join(" ", removeArticles).split(" ");
        String afterWhiteSpaceFix = String.join(" ", whiteSpaceFixs);
        String result = afterWhiteSpaceFix.toLowerCase(Locale.ENGLISH);
        return result;
    }

    float ExactMatchScore(List<String> groundTruths, String predict) {
        String[] groundTruth = groundTruths.get(0).replace("[\"", "").replace("\"]", "").split("\",\"");
        for(String groundTruthTmp: groundTruth) {
            if(Normalize(groundTruthTmp).equals(Normalize(predict))) {
                return 1.0f;
            }
        }
        return 0.0f;
    }


    float F1Score(List<String> groundTruths, String predict) {
        float finalScore = 0.0f;
        String[] predWords = predict.split(" ");
        Map<String, Integer> predMap = new HashMap<String, Integer>();
        for(String word: predWords) {
            String normalizeWord = Normalize(word);
            if(predMap.containsKey(normalizeWord)) {
                Integer counter = predMap.get(normalizeWord);
                predMap.put(normalizeWord, ++counter);
            }
            else {
                //Filter out the words the, an, a, etc
                if(Objects.equals(normalizeWord, "")){
                    continue;
                }
                predMap.put(normalizeWord, 1);
            }
        }

        String[] groundTruthsTmp = groundTruths.get(0).replace("[\"", "").replace("\"]", "").split("\",\"");
        for(String groundTruth: groundTruthsTmp) {
            String[] gtWords = groundTruth.split(" ");
            Map<String, Integer> gtMap = new HashMap<String, Integer>();
            for(String gtWord: gtWords) {
                String normalizedGtWord = Normalize(gtWord);
                if(gtMap.containsKey(normalizedGtWord)) {
                    Integer counter = gtMap.get(normalizedGtWord);
                    gtMap.put(normalizedGtWord, ++counter);
                }
                else {
                    //Filter out the words the, an, a, etc
                    if(Objects.equals(normalizedGtWord, "")){
                        continue;
                    }
                    gtMap.put(normalizedGtWord, 1);
                }
            }

            int numCommon = 0;
            for(String common: predMap.keySet()) {
                if(predMap.containsKey(common) && gtMap.containsKey(common)) {
                    numCommon += Math.min(predMap.get(common), gtMap.get(common));
                }
            }
            if(numCommon == 0) continue;

            float precison = 1.0f * numCommon / predMap.size();
            float recall = 1.0f * numCommon / gtMap.size();
            float f1 = (2 * precison * recall) / (precison + recall);
            finalScore = Math.max(f1, finalScore);
        }
        return finalScore;
    }



}
