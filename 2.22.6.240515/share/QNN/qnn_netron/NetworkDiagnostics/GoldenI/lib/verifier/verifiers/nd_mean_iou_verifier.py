# =============================================================================
#
#  Copyright (c) 2019-2021 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================
from pathlib import Path

import numpy as np
import pandas as pd

from lib.utils.nd_errors import get_message
from lib.utils.nd_exceptions import VerifierError, VerifierInputError
from lib.verifier.models import VerifyResult
from lib.verifier.verifiers.nd_base_verifier import BaseVerifier


class MeanIOUVerifier(BaseVerifier):
    TYPE = 'Mean IOU'
    UNITS = 'Mean IOU (%)'

    CLASS = 'classification'
    BOX = 'bbox'
    IOU = 'IOU'

    GOLDEN_CLASS = 'classification_golden'
    GOLDEN_BOX = 'bbox_golden'
    INF_CLASS = 'classification_inference'
    INF_BOX = 'bbox_inference'

    def __init__(self, background_classification=1.0):
        self.background_classification = background_classification
        self.data_frame = None

    @staticmethod
    def compute_iou(box1, box2):
        # type: (list, list) -> float
        """
        Compute the IOU for two boxes.
        'Boxes' are lists of size 4, where indices (3,2) represent the top-right corner and
        (1, 0) represent the bottom-left corner of the box.

        :param box1: list representing a box
        :param box2: list representing a box
        :return: IOU value as a float
        """
        # compute the area of intersection rectangle
        x_max = min(box1[3], box2[3])
        x_min = max(box1[1], box2[1])

        y_max = min(box1[2], box2[2])
        y_min = max(box1[0], box2[0])

        intersection_area = max(0, x_max - x_min) * max(0, y_max - y_min)

        expected_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        output_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

        # perfect intersection, avoid division by 0 and nan
        if intersection_area == expected_area and intersection_area == output_area:
            return 1.0

        iou = intersection_area / (expected_area + output_area - intersection_area)
        return iou

    @staticmethod
    def raw_data_to_data_frame(boxes_data, classes_data):
        # type: (np.array, np.array) -> pd.DataFrame
        return pd.DataFrame({
            MeanIOUVerifier.BOX: boxes_data.reshape(-1, 4).tolist(),
            MeanIOUVerifier.CLASS: classes_data
        })

    def process_boxes(self, g_boxes_data, g_classes_data, i_boxes_data, i_classes_data):
        # type: (np.array, np.array, np.array, np.array) -> pd.DataFrame
        def compute_iou_if_valid_classifications(series):
            # type: (pd.Series) -> float
            if series[self.GOLDEN_CLASS] != series[self.INF_CLASS]:
                return 0.0
            return MeanIOUVerifier.compute_iou(series[self.GOLDEN_BOX], series[self.INF_BOX])

        def join_golden_and_inference_dfs(golden_data_frame, inference_data_frame):
            # type: (pd.DataFrame, pd.DataFrame) -> pd.DataFrame
            """
            Merge a a golden DataFrame and an inference DataFrame horizontally by classifications. If there are more
            classification for either the golden or inference, then default to NaNs.

                ie.
                       IOU  boxes_golden  classification_golden  boxes_inference  classification_inference
                    0  NaN  [0, 0, 2, 2]          1               [0, 0, 3, 4]               1
                    1  NaN  [0, 0, 2, 2]          1                   nan                   nan

            :param golden_data_frame: DataFrame, formatted per MeanIOUVerifier.raw_data_to_data_frame
            :param inference_data_frame: DataFrame, formatted per MeanIOUVerifier.raw_data_to_data_frame
            :return: a DataFrame(columns=['IOU' 'boxes_golden', 'boxes_inference', 'classification_golden',
             'classification_inference'])
            """
            golden_classes = golden_data_frame[self.CLASS].unique().tolist()
            inference_classes = inference_data_frame[self.CLASS].unique().tolist()

            classes = set(golden_classes + inference_classes)
            # We should not be matching golden background to inference background classifications as they do not
            # affect the mean IOU
            classes.discard(self.background_classification)

            df = pd.DataFrame(columns=[self.GOLDEN_BOX, self.GOLDEN_CLASS, self.INF_BOX, self.INF_CLASS, self.IOU])
            for classification in classes:
                # reset indexes since we are taking a slice, and the indexes may not be in sequential order
                goldens = golden_df[golden_df[self.CLASS] == classification].reset_index(drop=True)
                inferences = inference_df[inference_df[self.CLASS] == classification].reset_index(drop=True)

                combined = pd.merge(goldens, inferences, left_index=True, right_index=True, how='outer',
                                    suffixes=['_golden', '_inference'])
                df = df.append(combined, sort=False)
            return df.reset_index(drop=True)

        golden_df = MeanIOUVerifier.raw_data_to_data_frame(g_boxes_data, g_classes_data)
        inference_df = MeanIOUVerifier.raw_data_to_data_frame(i_boxes_data, i_classes_data)
        joined_df = join_golden_and_inference_dfs(golden_df, inference_df)

        if not joined_df.empty:
            joined_df[self.IOU] = joined_df.apply(compute_iou_if_valid_classifications, axis=1)

        return joined_df

    def verify(self, layer_type, tensor_dimensions, golden, inference):

        def has_valid_shapes(expected_boxes, expected_classes, output_boxes, output_classes):
            # Each box is of size 4, and has exactly one classification
            return len(expected_boxes) == len(output_boxes) and \
                   len(expected_classes) == len(output_classes) and \
                   len(expected_boxes) / 4 == len(expected_classes) and \
                   len(output_boxes) / 4 == len(output_classes)

        def compute_mean_iou(df):
            # Consider true positives, false positives, and false negatives, but not true negatives for mean IOU
            true_negative = (df[self.GOLDEN_CLASS] == self.background_classification) & \
                            (df[self.INF_CLASS] == self.background_classification)
            mean = df.loc[~true_negative][self.IOU].mean()

            return mean if not np.isnan(mean) else 0.0

        if len(golden) != 2 or len(inference) != 2:
            raise VerifierInputError(get_message("ERROR_VERIFIER_MEAN_IOU_INCORRECT_INPUT_SIZE"))

        g_boxes, g_classes = golden
        i_boxes, i_classes = inference

        if not has_valid_shapes(g_boxes, g_classes, i_boxes, i_classes):
            raise VerifierError(get_message('ERROR_VERIFIER_MEAN_IOU_DIFFERENT_SIZE')(
                len(g_boxes), len(g_classes), len(i_boxes), len(i_classes)))

        # round our classifications since they should be whole numbers (albeit in float form)
        i_classes = np.rint(i_classes)

        self.data_frame = self.process_boxes(g_boxes, g_classes, i_boxes, i_classes)
        mean_iou = compute_mean_iou(self.data_frame)
        return VerifyResult(layer_type, g_boxes.shape, tensor_dimensions,
                            MeanIOUVerifier.TYPE, mean_iou, MeanIOUVerifier.UNITS)

    def _to_csv(self, path):
        if isinstance(self.data_frame, pd.DataFrame):
            path = Path(path).with_suffix('.csv')
            self.data_frame.to_csv(path, encoding='utf-8', index=False)

    def _to_html(self, path):
        if isinstance(self.data_frame, pd.DataFrame):
            path = Path(path).with_suffix('.html')

            # remove the column width limit so there is no truncation of values
            with pd.option_context('display.max_colwidth', -1):
                self.data_frame.to_html(path, classes='table', index=False)

    def save_data(self, path):
        self._to_csv(path)
        self._to_html(path)

    @staticmethod
    def validate_config(configs):
        params={}
        err_str="Unable to validate mean IOU. Expected format as background_classification <val>"
        if len(configs) % 2 != 0:
            return False, {'error':"Cannot pair verifier parameter.{}".format(err_str)}
        if len(configs) > 1:
            for i in range(0,len(configs),2):
                if configs[i] == 'background_classification':
                    if configs[i] not in params:
                        try:
                            params[configs[i]]=float(configs[i+1])
                        except ValueError:
                            return False,  {'error':"Can't convert data:{}.{}".format(configs[i+1],err_str)}
                else:
                    return False,  {'error':"Illegal parameter: {}.{}".format(configs[i],err_str)}
        return True, params