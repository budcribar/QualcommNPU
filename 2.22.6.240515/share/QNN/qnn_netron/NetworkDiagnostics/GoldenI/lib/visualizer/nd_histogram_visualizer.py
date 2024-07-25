# =============================================================================
#
#  Copyright (c) 2019 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================


import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt


class HistogramVisualizer:
    @staticmethod
    def visualize(inference_data, golden_data, dest):
        fig, (ax_1, ax_2) = plt.subplots(1, 2, sharex='col', sharey='row')
        ax_1.set_xlabel('Inference Tensor Value Range')
        ax_1.set_ylabel('Frequency')
        ax_2.set_xlabel('Golden Tensor Value Range')

        ax_1.hist(inference_data)
        ax_2.hist(golden_data, color='gold')
        plt.tight_layout()
        plt.savefig(dest)
        plt.close(fig)
