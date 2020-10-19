#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 18 23:39:18 2020

@author: zhuoyin94
"""

import os
import gc
from datetime import datetime
import warnings
import pandas as pd
import numpy as np
from tqdm import tqdm
from numba import njit
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from utils import LoadSave
import lightgbm as lgb

np.random.seed(2020)
warnings.filterwarnings('ignore')
sns.set(style="ticks", font_scale=1.2, palette='deep', color_codes=True)
colors = ["red"] + ["C{}".format(i) for i in range(9+1)]
###############################################################################
def load_pkl(file_name=None):
    """Loading *.pkl from the path .//cached_data//"""
    file_processor = LoadSave()
    return file_processor.load_data(path=".//cached_data//{}".format(file_name))


def load_csv(file_name=None, path_name=".//cached_data//", nrows=100):
    """Load the original *.csv data."""
    total_name = path_name + file_name
    csv_data = pd.read_csv(total_name, nrows=nrows)
    return csv_data


@njit
def njit_scan(df_index=None, df_time_index=None, df_feat_vals=None, lagging_mins=20):
    feat_vals_to_compute = []
    for curr_ind in df_index:
        # Look back feature values
        feat_vals_tmp = []
        curr_time, look_back_time = df_time_index[curr_ind], df_time_index[curr_ind]
        time_delta = lagging_mins * 60 * 1000
        while((curr_ind >= 0) and (abs(curr_time - look_back_time) <= time_delta)):
            feat_vals_tmp.append(df_feat_vals[curr_ind])
            curr_ind -= 1
            look_back_time = df_time_index[curr_ind]

        # Store the feat_vals
        feat_vals_to_compute.append(feat_vals_tmp)
    return feat_vals_to_compute


def compute_lagging_statistical_feats(df=None,
                                      feat_col_name=None,
                                      time_col_name="DOTTING_TIME",
                                      operation_list=None,
                                      lagging_mins=20):
    """Compute the lagging statistica features based on a specific time_range(lagging_mins)."""
    if feat_col_name is None:
        raise ValueError("The lagging statistical feature name is not specified !")
    if operation_list is None:
        operation_list = [np.mean]
    df_index = np.arange(len(df))
    df_time_index = df[time_col_name].values
    df_feat_vals = df[feat_col_name].values

    feat_df = pd.DataFrame(None)
    for fcn in operation_list:
        feat_name = fcn.__name__
        feat_name = feat_col_name + "_" + feat_name + "_{}_min".format(lagging_mins)

        feat_vals_to_compute = njit_scan(df_index, df_time_index,
                                         df_feat_vals, lagging_mins)
        feat_df[feat_name] = list(map(fcn, feat_vals_to_compute))

        # feat_res = []
        # for curr_ind in df_index:

        #     # Look back feature values
        #     feat_vals_tmp = []
        #     curr_time, look_back_time = df_time_index[curr_ind], df_time_index[curr_ind]
        #     time_delta = lagging_mins * 60 * 1000
        #     while((curr_ind >= 0) and (abs(curr_time - look_back_time) <= time_delta)):
        #         feat_vals_tmp.append(df_feat_vals[curr_ind])
        #         curr_ind -= 1
        #         look_back_time = df_time_index[curr_ind]

        #     # Compute the feature values
        #     try:
        #         feat_res.append(fcn(feat_vals_tmp))
        #     except:
        #         feat_res.append(np.nan)
        # feat_df[feat_name] = feat_res
    df = pd.concat([df, feat_df], axis=1)
    return df


# def compute_lagging_feats(df=None, 
#                           feat_col_name=None,
#                           time_col_name="DOTTING_TIME",
#                           lagging_mins_list=None):
#     """Lagging features for the feat in the df"""
#     lagging_mins_list = lagging_mins_list or [5, 10, 15, 20, 25]
#     df_index, df_time_index = np.arange(len(df)), df[time_col_name].values
#     feat_vals = df[feat_col_name].values

#     for lagging_mins in lagging_mins_list:
#         feat_res = []
#         for curr_ind in df_index:

#             # Look back feature values
#             feat_vals_tmp = []
#             curr_time, time_delta = df_time_index[curr_ind], lagging_mins * 60 * 1000
#             if ()

#     for step in lagging_steps:
#         df[name+"_lagging_{}".format(step)] = df[name].shift(step)
#     return df


def trend(n):
    """https://stackoverflow.com/questions/17578115/pass-percentiles-to-pandas-agg-function"""
    def trend_(x):
        if len(x) <= 1:
            return np.nan
        return x[-1] - x[0]
    trend_.__name__ = 'trend_%s' % n
    return trend_


if __name__ == "__main__":
    # Loading all the data
    # ----------------------------
    submission = pd.read_csv(".//data//submit_example.csv")
    total_df = load_pkl("total_df.pkl")
    train_df = total_df[total_df["ID"].isnull()].reset_index(drop=True)
    test_df = total_df[total_df["ID"].notnull()].reset_index(drop=True)

    key_cols = ["ID", "QUEUE_ID", "DOTTING_TIME"]
    cat_cols = ["STATUS", "QUEUE_TYPE", "PLATFORM", "RESOURCE_TYPE"]
    numeric_cols = ["CU", "CPU_USAGE", "MEM_USAGE", "LAUNCHING_JOB_NUMS",
                    "RUNNING_JOB_NUMS", "SUCCEED_JOB_NUMS", "CANCELLED_JOB_NUMS",
                    "FAILED_JOB_NUMS", "DISK_USAGE"]
    feat_df = total_df[key_cols].copy()

    # Feature engineering
    # ----------------------------
    for feat_name in tqdm(numeric_cols):
        tmp_df = compute_lagging_statistical_feats(
            df=total_df[key_cols+[feat_name]].copy(),
            feat_col_name=feat_name,
            operation_list=[np.mean, np.sum, np.median, np.std, np.ptp],
            lagging_mins=20)
        feat_df = pd.merge(feat_df, tmp_df, how="left", on=key_cols)

    # df = test_df.query("((QUEUE_ID == 4)) & ((ID == 820) | (ID == 2247) | (ID == 653))")
    # tmp = compute_lagging_feats(df=df.reset_index(drop=True),
    #                             feat_col_name="DISK_USAGE",
    #                             operation_list=[np.mean, np.sum],
    #                             lagging_mins=20)
