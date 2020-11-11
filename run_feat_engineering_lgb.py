#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  6 23:21:12 2020

@author: zhuoyin94
"""

import os, gc, warnings
from datetime import datetime
import pandas as pd
import numpy as np
from tqdm import tqdm
from numba import njit
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from utils import LoadSave, custom_metric
from sklearn.model_selection import StratifiedKFold, KFold
import lightgbm as lgb

np.random.seed(2022)
warnings.filterwarnings('ignore')
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


def nearby(n):
    def nearby_(x):
        try:
            return x[n]
        except:
            return np.nan
    nearby.__name__ = "nearby_{}".format(n)
    return nearby_


def trend(n):
    """https://stackoverflow.com/questions/17578115/pass-percentiles-to-pandas-agg-function"""
    def trend_(x):
        if len(x) <= 1:
            return np.nan
        return x[-1] - x[0]
    trend_.__name__ = 'trend_%s' % n
    return trend_


@njit
def njit_scan(df_index=None, df_time_index=None, df_feat_vals=None, lagging_mins=20):
    """Finding the data points within the lagging_mins range"""
    feat_vals_to_compute = []
    for curr_ind in df_index:
        # Look back for feature values
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


def compute_lagging_statistical_feats(df=None, feat_col_name=None,
                                      time_col_name="DOTTING_TIME",
                                      operation_list=None, lagging_mins=20):
    """Compute the lagging statistica features based on a specific time_range(lagging_mins)(Maximum == 20)."""
    if feat_col_name is None:
        raise ValueError("The lagging statistical feature name is not specified !")
    if operation_list is None:
        operation_list = [np.mean]
    df_index = np.arange(len(df))
    df_time_index = df[time_col_name].values
    df_feat_vals = df[feat_col_name].values

    # Scan the value according to the window
    feat_vals_to_compute = njit_scan(df_index, df_time_index,
                                     df_feat_vals, lagging_mins)

    feat_df = pd.DataFrame(None)
    for fcn in operation_list:
        feat_name = fcn.__name__
        feat_name = feat_col_name + "_" + feat_name + "_{}".format(lagging_mins)
        feat_df[feat_name] = list(map(fcn, feat_vals_to_compute))
    df = pd.concat([df, feat_df], axis=1)

    return df


if __name__ == "__main__":
    # Loading all the data
    # ----------------------------
    print("\n")
    total_df = load_pkl("total_df.pkl")
    train_df = total_df[total_df["ID"].isnull()].reset_index(drop=True)
    test_df = total_df[total_df["ID"].notnull()].reset_index(drop=True)

    key_cols = ["ID", "QUEUE_ID", "DOTTING_TIME"]
    cat_cols = ["STATUS", "QUEUE_TYPE", "PLATFORM", "RESOURCE_TYPE", "CU"]
    numeric_cols = ["CPU_USAGE", "MEM_USAGE", "LAUNCHING_JOB_NUMS",
                    "RUNNING_JOB_NUMS", "SUCCEED_JOB_NUMS", "CANCELLED_JOB_NUMS",
                    "FAILED_JOB_NUMS", "DISK_USAGE"]
    feat_df = total_df[key_cols].copy()


    # Statistical Feature engineering
    # ----------------------------
    for feat_name in tqdm(numeric_cols):
        tmp_df = compute_lagging_statistical_feats(
            df=total_df[["DOTTING_TIME"]+[feat_name]].copy(),
            feat_col_name=feat_name,
            operation_list=[np.mean, np.ptp, np.std, np.sum, np.min, np.max, nearby(1), nearby(2)],
            lagging_mins=20)
        feat_df = pd.concat([feat_df, tmp_df.drop(["DOTTING_TIME"], axis=1)], axis=1)

    for feat_name in cat_cols:
        if feat_name == "CU":
            continue
        tmp_df = pd.get_dummies(total_df[feat_name])
        tmp_df.columns = [feat_name + "_(" + str(item) + ")" for item in tmp_df.columns]
        feat_df = pd.concat([feat_df, tmp_df], axis=1)
    feat_df = pd.concat([feat_df, total_df[["CU"]]], axis=1)


    # Create targets dataframe
    # ----------------------------
    maximum_tol_mins = 4
    target_df = train_df[["QUEUE_ID", "DOTTING_TIME"]].copy()
    target_feat_cols = ["CPU_USAGE", "LAUNCHING_JOB_NUMS"]
    forward_steps_list = [1, 2, 3, 4, 5]

    for feat_name in target_feat_cols:
        for step in forward_steps_list:
            tol_time_diff = step * 5 * 60 * 1000 + maximum_tol_mins * 60 * 1000
            target_feat_name = feat_name + "_" + str(step)
            tmp_df = train_df[["DOTTING_TIME", "QUEUE_ID", feat_name]].copy()
            tmp_df = tmp_df.shift(periods=-step, axis=0).reset_index(drop=True)

            tmp_df.rename({"DOTTING_TIME": "DOTTING_TIME_SHIFT",
                            "QUEUE_ID": "QUEUE_ID_SHIFT",
                            feat_name: target_feat_name}, axis=1, inplace=True)
            target_df = pd.concat([target_df, tmp_df], axis=1)

            # Exclude invalid target values
            target_df[target_feat_name][target_df["QUEUE_ID"] != target_df["QUEUE_ID_SHIFT"]]= np.nan
            target_df[target_feat_name][abs(target_df["DOTTING_TIME"] - target_df["DOTTING_TIME_SHIFT"]) > tol_time_diff] = np.nan

            # Drop tmp columns
            target_df.drop(["DOTTING_TIME_SHIFT", "QUEUE_ID_SHIFT"],
                            axis=1, inplace=True)


    # Save feat engineering results
    # ----------------------------
    file_processor = LoadSave()
    total_results = [feat_df, target_df]
    file_processor.save_data(path=".//cached_data//{}.pkl".format("lgb_dense_feat"),
                             data=total_results)
