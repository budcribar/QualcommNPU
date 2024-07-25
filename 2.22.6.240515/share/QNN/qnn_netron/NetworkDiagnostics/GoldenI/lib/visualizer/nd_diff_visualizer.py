# =============================================================================
#
#  Copyright (c) 2021 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================


import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt


class DiffVisualizer:
    @staticmethod
    def visualize(inference_data, golden_data, dest):
        fig = plt.figure()
        # plot diff
        diff = inference_data - golden_data
        ax_1 = plt.subplot(2,1,1)
        ax_1.set_xlabel('Position')
        ax_1.set_ylabel('Value')
        ax_1.set_title("Diff between golden and inference data")
        plt.plot(range(diff.shape[0]), diff, label='Diff', color='red')

        # plot inference data
        ax_2 = plt.subplot(2,2,3)
        ax_2.set_xlabel('Position')
        ax_2.set_ylabel('Value')
        ax_2.set_title("Golden data")
        plt.plot(range(inference_data.shape[0]), inference_data, "*", color='blue')

        # plot golden data
        ax_3 = plt.subplot(2,2,4)
        ax_3.set_xlabel('Position')
        ax_3.set_ylabel('Value')
        ax_3.set_title("Inference data")
        plt.plot(range(golden_data.shape[0]), golden_data, "*", color='green')

        plt.tight_layout()
        plt.savefig(dest, figure=fig)
        plt.close(fig)
