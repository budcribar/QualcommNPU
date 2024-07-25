# =============================================================================
#
#  Copyright (c) 2023 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use('agg')
import matplotlib.pyplot as plt


class Visualizers:

    @staticmethod
    def histogram_visualizer(golden_data, target_data, dest):
        fig, (ax_1, ax_2) = plt.subplots(1, 2, sharex='col', sharey='row')
        ax_1.set_xlabel('Golden Tensor Value Range')
        ax_1.set_ylabel('Frequency')
        ax_2.set_xlabel('Target Tensor Value Range')

        ax_1.hist(golden_data, color='gold')
        ax_2.hist(target_data, color='blue')
        plt.tight_layout()
        plt.savefig(dest)
        plt.close(fig)

    @staticmethod
    def diff_visualizer(golden_data, target_data, dest):
        fig = plt.figure()
        # plot diff
        diff = target_data - golden_data
        ax_1 = plt.subplot(2, 1, 1)
        ax_1.set_xlabel('Position')
        ax_1.set_ylabel('Value')
        ax_1.set_title("Diff between golden and target data")
        plt.plot(range(diff.shape[0]), diff, label='Diff', color='red')

        # plot target data
        ax_2 = plt.subplot(2, 2, 3)
        ax_2.set_xlabel('Position')
        ax_2.set_ylabel('Value')
        ax_2.set_title("Golden data")
        plt.plot(range(golden_data.shape[0]), golden_data, "*", color='gold')

        # plot golden data
        ax_3 = plt.subplot(2, 2, 4)
        ax_3.set_xlabel('Position')
        ax_3.set_ylabel('Value')
        ax_3.set_title("Target data")
        plt.plot(range(target_data.shape[0]), target_data, "*", color='blue')

        plt.tight_layout()
        plt.savefig(dest)
        plt.close(fig)

    @staticmethod
    def cdf_visualizer(golden_data, target_data, dest):

        golden_data_hist, golden_data_edges = np.histogram(golden_data, 256)
        golden_data_centers = (golden_data_edges[:-1] + golden_data_edges[1:]) / 2
        golden_data_cdf = np.cumsum(golden_data_hist / golden_data_hist.sum())

        target_data_hist, target_data_edges = np.histogram(target_data, 256)
        target_data_centers = (target_data_edges[1:] + target_data_edges[:-1]) / 2
        target_data_cdf = np.cumsum(target_data_hist / target_data_hist.sum())

        plt.figure(figsize=[20, 15])
        plt.plot(golden_data_centers, golden_data_cdf, color='gold')
        plt.plot(target_data_centers, target_data_cdf, color='blue')
        plt.legend(["Golden data CDF", "Target data CDF"])
        plt.savefig(dest)
        plt.close()

    @staticmethod
    def distribution_visualizer(data, dest, target_min, target_max, calibrated_min, calibrated_max):

        plt.figure()
        pd.DataFrame(data, columns=['Target data distribution']).plot(kind='kde', color='blue')

        plt.axvline(x=target_min, color='red')
        plt.text(target_min, 0, f'{target_min:.2f}', rotation=90)

        plt.axvline(x=target_max, color='red')
        plt.text(target_max, 0, f'{target_max:.2f}', rotation=90)

        plt.axvline(x=calibrated_min, color='green')
        plt.text(calibrated_min, 0, f'{calibrated_min:.2f}', rotation=90)

        plt.axvline(x=calibrated_max, color='green')
        plt.text(calibrated_max, 0, f'{calibrated_max:.2f}', rotation=90)

        target_data_distribution = matplotlib.patches.Patch(color='blue',
                                                            label='Target data distribution')
        target_hightlight = matplotlib.patches.Patch(color='red', label='Target min/max')
        calibrated_hightlight = matplotlib.patches.Patch(color='green', label='Calibrated min/max')
        plt.legend(handles=[target_data_distribution, target_hightlight, calibrated_hightlight])

        plt.savefig(dest)
        plt.close()
