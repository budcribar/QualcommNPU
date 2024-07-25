# =============================================================================
#
#  Copyright (c) 2021 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================
from typing import Union, Tuple

import pandas as pd


class VerifierAnalyzer:
    """A Class to identify primary sources of verifier error given summary dataframe."""

    @staticmethod
    def get_metric(verifier, df, tensor_name):
        # type: (str, pd.DataFrame, str) -> Union[float, None]
        """Returns verifier metric for specific tensor name."""
        try:
            return float(df.loc[df['Name'] == tensor_name][verifier].values[0])
        except:
            return None

    @staticmethod
    def analyze(verifier, *args, **kwargs):
        """Call analyzers based on verifier type."""

        if 'threshold' in kwargs and kwargs['threshold'] is None:
            kwargs.pop('threshold')

        problem_tensor_name = None
        metric_val = None
        try:
            if verifier == 'CosineSimilarity':
                problem_tensor_name, metric_val = VerifierAnalyzer.cosine_similarity_analyze(*args, **kwargs)
            elif verifier == 'RtolAtol':
                problem_tensor_name, metric_val = VerifierAnalyzer.rtol_atol_analyze(*args, **kwargs)
            elif verifier == 'AdjustedRtolAtol':
                problem_tensor_name, metric_val = VerifierAnalyzer.adjusted_rtol_atol_analyze(*args, **kwargs)
            elif verifier == 'TopK':
                problem_tensor_name, metric_val = VerifierAnalyzer.topk_analyze(*args, **kwargs)
            elif verifier == 'L1Error':
                problem_tensor_name, metric_val = VerifierAnalyzer.l1error_analyze(*args, **kwargs)
            elif verifier == 'SQNR':
                problem_tensor_name, metric_val = VerifierAnalyzer.sqnr_analyze(*args, **kwargs)
            elif verifier == 'MSE':
                problem_tensor_name, metric_val = VerifierAnalyzer.mse_analyze(*args, **kwargs)
        except TypeError:
            raise NotImplementedError("'threshold' keyword argument not accepted by Verifier Analyzer")

        return problem_tensor_name, metric_val

    @staticmethod
    def is_change_significant(verifier, *args, **kwargs):
        """Call analyzers based on verifier type."""

        is_significant = None
        if verifier == 'CosineSimilarity':
            is_significant = VerifierAnalyzer.cosine_similarity_change_significant(*args, **kwargs)
        elif verifier == 'RtolAtol':
            is_significant = VerifierAnalyzer.rtol_atol_change_significant(*args, **kwargs)
        elif verifier == 'AdjustedRtolAtol':
            is_significant = VerifierAnalyzer.adjusted_rtol_atol_change_significant(*args, **kwargs)
        elif verifier == 'TopK':
            is_significant = VerifierAnalyzer.topk_change_significant(*args, **kwargs)
        elif verifier == 'L1Error':
            is_significant = VerifierAnalyzer.l1error_change_significant(*args, **kwargs)
        elif verifier == 'SQNR':
            is_significant = VerifierAnalyzer.sqnr_change_significant(*args, **kwargs)
        elif verifier == 'MSE':
            is_significant = VerifierAnalyzer.mse_change_significant(*args, **kwargs)
        return is_significant

    @staticmethod
    def l1error_analyze(df):
        # type: (pd.DataFrame) -> Tuple[str, float]
        """Determine problematic tensor in dataframe based on L1 Error/MAE"""
        # Assume splitting at max val
        problem_tensor_idx = df["L1Error"].idxmax()
        problem_tensor_name = df.iloc[problem_tensor_idx]["Name"]
        return problem_tensor_name

    @staticmethod
    def mse_analyze(df):
        # type: (pd.DataFrame) -> Tuple[str, float]
        """Determine problematic tensor in dataframe based on MSE"""
        # Assume splitting at max val
        problem_tensor_idx = df["MSE"].idxmax()
        problem_tensor_name = df.iloc[problem_tensor_idx]["Name"]
        return problem_tensor_name, df.iloc[problem_tensor_idx]["MSE"]

    @staticmethod
    def rtol_atol_analyze(df):
        # type: (pd.DataFrame) -> Tuple[str, float]
        """Determine problematic tensor in dataframe based on rtol atol"""
        raise NotImplementedError

    @staticmethod
    def adjusted_rtol_atol_analyze(df):
        # type: (pd.DataFrame) -> Tuple[str, float]
        """Determine problematic tensor in dataframe based on adjusted rtol atol"""
        raise NotImplementedError

    @staticmethod
    def topk_analyze(df):
        # type: (pd.DataFrame) -> Tuple[str, float]
        """Determine problematic tensor in dataframe based on topk"""
        raise NotImplementedError

    @staticmethod
    def cosine_similarity_analyze(df, threshold=0.85):
        # type: (pd.DataFrame, float) -> Tuple[str, float]
        """Determine problematic tensor in dataframe based on Cosine Similarity"""
        # Assume df is ordered
        # higher is better

        # problem_tensor_idx = df["CosineSimilarity"].idxmin()
        # problem_tensor_name = df.iloc[problem_tensor_idx]["Name"]
        # print(problem_tensor_name, df["CosineSimilarity"][problem_tensor_idx])
        # return problem_tensor_name

        for _,row in df.iterrows():
            if row["CosineSimilarity"] < threshold:
                return row["Name"], row["CosineSimilarity"]
        return None, None

        # df_prev = df["CosineSimilarity"].shift(1)
        # p_changes = (df_prev - df["CosineSimilarity"]) / df_prev # check > 20%?
        # idx = p_changes.idxmax()
        # print(df.iloc[idx]["Name"], df_prev.iloc[idx], p_changes.iloc[idx], df.iloc[idx]["CosineSimilarity"])
        # return df.iloc[idx]["Name"], df.iloc[idx]["CosineSimilarity"]

    @staticmethod
    def sqnr_analyze(df, threshold=0.85):
        # type: (pd.DataFrame, float) -> Tuple[str, float]
        """Determine problematic tensor in dataframe based on SQNR"""
        # Assume df is ordered
        # higher is better

        # problem_tensor_idx = df["SQNR"].idxmin()
        # problem_tensor_name = df.iloc[problem_tensor_idx]["Name"]
        # print(problem_tensor_name, df["SQNR"][problem_tensor_idx])
        # return problem_tensor_name

        # df_prev = df["SQNR"].shift(1)
        # p_changes = (df_prev - df["SQNR"]) / df_prev
        # idx = p_changes.idxmax()
        # return df.iloc[idx]["Name"], df.iloc[idx]["SQNR"]

        df_num = pd.to_numeric(df["SQNR"], errors='coerce')
        df_prev = pd.to_numeric(df["SQNR"].shift(1), errors='coerce')
        for idx, (name, p_of_last) in enumerate(zip(df["Name"], df_num / df_prev)):
            if p_of_last < threshold:
                return name, df_num.iloc[idx]
        return None, None

    @staticmethod
    def l1error_change_significant(df, original, actual):
        # type: (pd.DataFrame, float, float) -> bool
        """Determine if change/error is significant based on L1Error"""
        raise NotImplementedError

    @staticmethod
    def mse_change_significant(df, original, actual):
        # type: (pd.DataFrame, float, float) -> bool
        """Determine if change/error is significant based on MSE"""
        raise NotImplementedError

    @staticmethod
    def rtol_atol_change_significant(df, original, actual):
        # type: (pd.DataFrame, float, float) -> bool
        """Determine if change/error is significant based on rtol atol"""
        raise NotImplementedError

    @staticmethod
    def adjusted_rtol_atol_change_significant(df, original, actual):
        # type: (pd.DataFrame, float, float) -> bool
        """Determine if change/error is significant based on ajusted rtol atol"""
        raise NotImplementedError

    @staticmethod
    def topk_change_significant(df, original, actual):
        # type: (pd.DataFrame, float, float) -> bool
        """Determine if change/error is significant based on TopK"""
        raise NotImplementedError

    @staticmethod
    def cosine_similarity_change_significant(df, original, actual):
        # type: (pd.DataFrame, float, float) -> bool
        """Determine if change/error is significant based on Cosine Similarity"""
        if actual < (1 + original) / 2:
            return True
        else:
            return False

    @staticmethod
    def sqnr_change_significant(df, original, actual):
        # type: (pd.DataFrame, float, float) -> bool
        """Determine if change/error is significant based on SQNR"""
        if actual < 1.5 * original:
            return True
        else:
            return False
