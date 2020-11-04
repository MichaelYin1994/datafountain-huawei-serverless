#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 28 20:29:34 2019

@author: yinzhuo
"""

import os
import time
import pickle
import warnings
from datetime import datetime
from functools import wraps
import matplotlib.pyplot as plt
import pandas as pd
from numba import jit
from sklearn.metrics import precision_score, f1_score, recall_score, accuracy_score, precision_recall_curve, auc
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import StratifiedKFold, KFold
import numpy as np
from scipy import sparse
from scipy.signal import savgol_filter
from numpy import iinfo, finfo, int8, int16, int32, int64, float32, float64
import lightgbm as lgb
import xgboost as xgb

warnings.filterwarnings('ignore')
###############################################################################
###############################################################################
def timefn(fcn):
    """Decorator for efficency analysis. """
    @wraps(fcn)
    def measure_time(*args, **kwargs):
        start = time.time()
        result = fcn(*args, **kwargs)
        end = time.time()
        print("@timefn: " + fcn.__name__ + " took {:.5f}".format(end-start)
            + " seconds.")
        return result
    return measure_time


def custom_metric(y_true, y_pred):
    y_sub = y_true - y_pred
    y_max = np.max([y_true, y_pred], axis=0)

    y_res = np.zeros(y_sub.shape[0])
    y_res[y_max != 0] = y_sub[y_max != 0] / y_max[y_max != 0]
    return np.abs(y_res)


def plot_metric(history=None, metric_type="acc", **kwargs):
    """Plot the training curve of Tensorflow NN"""
    train_metric = history.history[metric_type]
    valid_metric = history.history["val_"+metric_type]
    epochs = list(range(1, len(train_metric)+1))

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(epochs, train_metric, color="green", marker="o", linestyle=" ",
            markersize=3, label="train_{}".format(metric_type))
    ax.plot(epochs, valid_metric, color="k", marker="s", linestyle=" ",
            markersize=3, label="valid_{}".format(metric_type))

    is_plot_smoothed_curve = kwargs.pop("is_plot_smoothed_curve", True)
    sg_window_length = kwargs.pop("sg_window_length", 7)
    sg_polyorder = kwargs.pop("sg_polyorder", 3)
    if is_plot_smoothed_curve:
        train_metric_smoothed = savgol_filter(train_metric,
                                              window_length=sg_window_length,
                                              polyorder=sg_polyorder)
        valid_metric_smoothed = savgol_filter(valid_metric,
                                              window_length=sg_window_length,
                                              polyorder=sg_polyorder)

        ax.plot(epochs, train_metric_smoothed, color="blue", linestyle="-",
                linewidth=1.5, label="train_smoothed_{}".format(metric_type))
        ax.plot(epochs, valid_metric_smoothed, color="r", linestyle="-",
                linewidth=1.5, label="valid_smoothed_{}".format(metric_type))

    ax.tick_params(axis="both", labelsize=10)
    ax.set_xlim(1, len(epochs)+1)
    ax.set_xlabel("Epochs", fontsize=10)
    ax.set_ylabel(metric_type, fontsize=10)
    ax.set_title("#Epochs: {}".format(len(epochs)), fontsize=10)
    ax.legend(fontsize=10)
    ax.grid(True)
    plt.tight_layout()


@timefn
def basic_feature_report(data_table=None, precent=None):
    """Reporting basic characteristics of the tabular data data_table."""
    precent = precent or [0.01, 0.25, 0.5, 0.75, 0.95, 0.9995]
    if data_table is None:
        return None
    num_samples = len(data_table)

    # Basic statistics
    basic_report = data_table.isnull().sum()
    basic_report = pd.DataFrame(basic_report, columns=["#missing"])
    basic_report["missing_precent"] = basic_report["#missing"]/num_samples
    basic_report["#uniques"] = data_table.nunique(dropna=False).values
    basic_report["types"] = data_table.dtypes.values
    basic_report.reset_index(inplace=True)
    basic_report.rename(columns={"index":"feature_name"}, inplace=True)

    # Basic quantile of data
    data_description = data_table.describe(precent).transpose()
    data_description.reset_index(inplace=True)
    data_description.rename(columns={"index":"feature_name"}, inplace=True)
    basic_report = pd.merge(basic_report, data_description,
        on='feature_name', how='left')
    return basic_report


class LoadSave():
    """Class for loading and saving object in .pkl format."""
    def __init__(self, file_name=None):
        self._file_name = file_name

    def save_data(self, data=None, path=None):
        """Save data to path."""
        if path is None:
            assert self._file_name is not None, "Invaild file path !"
        else:
            self._file_name = path
        self.__save_data(data)

    def load_data(self, path=None):
        """Load data from path."""
        if path is None:
            assert self._file_name is not None, "Invaild file path !"
        else:
            self._file_name = path
        return self.__load_data()

    def __save_data(self, data=None):
        """Save data to path."""
        print("--------------Start saving--------------")
        print("@SAVING DATA TO {}.".format(self._file_name))
        with open(self._file_name, 'wb') as file:
            pickle.dump(data, file)
        print("@SAVING SUCESSED !")
        print("----------------------------------------\n")

    def __load_data(self):
        """Load data from path."""
        if not self._file_name:
            raise ValueError("Invaild file path !")
        print("--------------Start loading--------------")
        print("@LOADING DATA FROM {}.".format(self._file_name))
        with open(self._file_name, 'rb') as file:
            data = pickle.load(file)
        print("@LOADING SUCCESSED !")
        print("-----------------------------------------\n")
        return data


class ReduceMemoryUsage():
    """
    ----------
    Author: Michael Yin
    E-Mail: zhuoyin94@163.com
    ----------
    
    @Description:
    ----------
    Reduce the memory usage of pandas dataframe.
    
    @Parameters:
    ----------
    data: pandas DataFrame-like
        The dataframe that need to be reduced memory usage.
    verbose: bool
        Whether to print the memory reduction information or not.
        
    @Return:
    ----------
    Memory-reduced dataframe.
    """
    def __init__(self, data_table=None, verbose=True):
        self._data_table = data_table
        self._verbose = verbose

    def type_report(self, data_table):
        """Reporting basic characteristics of the tabular data data_table."""
        data_types = list(map(str, data_table.dtypes.values))
        basic_report = pd.DataFrame(data_types, columns=["types"])
        basic_report["feature_name"] = list(data_table.columns)
        return basic_report

    @timefn
    def reduce_memory_usage(self):
        memory_reduced_data = self.__reduce_memory()
        return memory_reduced_data

    def __reduce_memory(self):
        print("\nReduce memory process:")
        print("-------------------------------------------")
        memory_before_reduced = self._data_table.memory_usage(
            deep=True).sum() / 1024**2
        types = self.type_report(self._data_table)
        if self._verbose is True:
            print("@Memory usage of data is {:.5f} MB.".format(
                memory_before_reduced))

        # Scan each feature in data_table, reduce the memory usage for features
        for ind, name in enumerate(types["feature_name"].values):
            # ToBeFixed: Unstable query.
            feature_type = str(
                types[types["feature_name"] == name]["types"].iloc[0])

            if (feature_type in "object") and (feature_type in "datetime64[ns]"):
                try:
                    feature_min = self._data_table[name].min()
                    feature_max = self._data_table[name].max()

                    # np.iinfo for reference:
                    # https://docs.scipy.org/doc/numpy/reference/generated/numpy.iinfo.html
                    # numpy data types reference:
                    # https://wizardforcel.gitbooks.io/ts-numpy-tut/content/3.html
                    if "int" in feature_type:
                        if feature_min > iinfo(int8).min and feature_max < iinfo(int8).max:
                            self._data_table[name] = self._data_table[name].astype(int8)
                        elif feature_min > iinfo(int16).min and feature_max < iinfo(int16).max:
                            self._data_table[name] = self._data_table[name].astype(int16)
                        elif feature_min > iinfo(int32).min and feature_max < iinfo(int32).max:
                            self._data_table[name] = self._data_table[name].astype(int32)
                        else:
                            self._data_table[name] = self._data_table[name].astype(int64)
                    else:
                        if feature_min > finfo(float32).min and feature_max < finfo(float32).max:
                            self._data_table[name] = self._data_table[name].astype(float32)
                        else:
                            self._data_table[name] = self._data_table[name].astype(float64)
                except Exception as error_msg:
                    print("\n--------ERROR INFORMATION---------")
                    print(error_msg)
                    print("Error on the {}".format(name))
                    print("--------ERROR INFORMATION---------\n")
            if self._verbose is True:
                print("Processed {} feature({}), total is {}.".format(
                    ind + 1, name, len(types)))

        memory_after_reduced = self._data_table.memory_usage(
            deep=True).sum() / 1024**2
        if self._verbose is True:
            print("@Memory usage after optimization: {:.5f} MB.".format(
                memory_after_reduced))
            print("@Decreased by {:.5f}%.".format(
                100 * (memory_before_reduced - memory_after_reduced) / memory_before_reduced))
        print("-------------------------------------------")
        return self._data_table


def pred_to_submission(y_valid=None, y_pred=None, score=None, save_oof=False,
                       sub_str_field="", best_threshold=0.5):
    """Save the oof prediction results to the local path."""
    sub_ind = len(os.listdir(".//submissions//")) + 1
    file_name = "{}_{}_tf1_{}_tacc_{}_vf1_{}_vacc_{}".format(
        sub_ind, sub_str_field,
        str(round(score["train_f1"].mean(), 4)).split(".")[1],
        str(round(score["train_acc"].mean(), 4)).split(".")[1],
        str(round(score["valid_f1"].mean(), 4)).split(".")[1],
        str(round(score["valid_acc"].mean(), 4)).split(".")[1])

    # Saving the submissions.
    submission = pd.DataFrame(None)
    submission["id"], submission["reply_id"] = y_pred["id"], y_pred["reply_id"]
    submission["target"] = np.where(y_pred.drop(["id", "reply_id"], axis=1).values[:, 1] > best_threshold,
                                    1, 0)
    submission.to_csv(".//submissions//{}.tsv".format(file_name),
                      header=False, index=False, encoding="utf-8",
                      sep="\t")

    # Submission stat
    print("\n---------------------")
    print("[SUB] Saving {} to the local.".format(file_name))
    pos_precent = len(submission.query("target == 1"))/len(submission) * 100
    neg_precent = len(submission.query("target == 0"))/len(submission) * 100
    print("[SUB] Submission match precent(1): {:.5f}%, not match precent(0): {:.5f}%.".format(
        pos_precent, neg_precent))
    print("---------------------")

    # Saving the oof scores.
    if save_oof:
        y_valid.to_csv(".//submission_oof//{}_valid.csv".format(file_name),
                       index=False, encoding="utf-8")
        y_pred.to_csv(".//submission_oof//{}_pred.csv".format(file_name),
                      index=False, encoding="utf-8")

    # Saving the oof scores.
    if save_oof:
        y_valid.to_csv(".//submission_oof//{}_valid.csv".format(file_name),
                       index=False, encoding="utf-8")
        y_pred.to_csv(".//submission_oof//{}_pred.csv".format(file_name),
                      index=False, encoding="utf-8")


def search_best_thresold_f1(y_pred_proba, y_true):
    """Searching for the best f1 thresold."""
    best_f1, best_threshold = 0, 0
    for threshold in range(20, 90):
        thresold = threshold / 100
        y_pred_label = np.where(y_pred_proba > thresold, 1, 0)
        score_tmp = f1_score(y_true.reshape(-1, 1), y_pred_label.reshape(-1, 1))
        if score_tmp > best_f1:
            best_f1 = score_tmp
            best_threshold = threshold
    return best_f1, best_threshold/100


@jit
def compute_longest_common_subsequence(seq_x=None, seq_y=None, max_pos_diff=3):
    """Calculate LCSS."""
    # Basic stats
    length_x, length_y = len(seq_x), len(seq_y)
    norm_factor = min(length_x, length_y)

    # Early stop conditions
    if length_x == 0 or length_y == 0:
        return 0
    elif length_x == 1:
        if seq_x[0] in seq_y:
            return 1/norm_factor
        else:
            return 0
    elif length_y == 1:
        if seq_y[0] in seq_x:
            return 1/norm_factor
        else:
            return 0

    # Dynamic programming for calculating LCSS
    dp = np.zeros((length_x+1, length_y+1))
    for i in range(1, length_x+1):
        for j in range(1, length_y+1):
            pos_diff = abs(i - j)
            if (pos_diff <= max_pos_diff) and (seq_x[i-1] == seq_y[j-1]):
                dp[i, j] = dp[i-1, j-1] + 1
            else:
                dp[i, j] = max(dp[i-1, j], dp[i, j-1])
    return dp[-1, -1]/norm_factor


@jit
def compute_edit_distance(seq_x=None, seq_y=None):
    """Calculate edit distance."""
    # Basic stats
    length_x, length_y = len(seq_x), len(seq_y)
    norm_factor = length_x + length_y

    # Early stop conditions
    if length_x == 0 or length_y == 0:
        return max(len(seq_x), len(seq_y))

    # Initializing the dp mat
    dp = np.zeros((length_x+1, length_y+1))
    dp[0, :] = np.arange(0, length_y+1)
    dp[:, 0] = np.arange(0, length_x+1)

    # Dynamic programming
    for i in range(1, length_x+1):
        for j in range(1, length_y+1):
            if seq_x[i-1] == seq_y[j-1]:
                subcost = 0
            else:
                subcost = 2
            dp[i, j] = min(dp[i-1][j]+1, dp[i][j-1]+1, dp[i-1][j-1]+subcost)
    return dp[i, j]/norm_factor
