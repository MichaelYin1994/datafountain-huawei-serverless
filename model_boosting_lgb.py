#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 18 23:39:18 2020

@author: zhuoyin94
"""

import os, gc, warnings
from datetime import datetime
import pandas as pd
import numpy as np
from tqdm import tqdm
from numba import njit
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from utils import LoadSave, custom_metric
from sklearn.model_selection import StratifiedKFold, KFold
import lightgbm as lgb

np.random.seed(2021)
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
def njit_scan(df_index=None,
              df_time_index=None,
              df_feat_vals=None,
              lagging_mins=20):
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


def compute_lagging_statistical_feats(df=None, feat_col_name=None,
                                      time_col_name="DOTTING_TIME", operation_list=None,
                                      lagging_mins=20):
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


if __name__ == "__main__":
    # Pre-setting global parameters:
    # ----------------------------
    N_FOLDS = 5
    RANDOM_SEED = 2021
    EARLY_STOP_ROUNDS = 100
    lgb_params = {"boosting_type": "gbdt",
                  "objective": "regression",
                  "metric": {"l1"},
                  "n_estimators": 2000,
                  "num_leaves": 31,
                  "max_depth": 4,
                  "learning_rate": 0.07,
                  "colsample_bytree": 0.95,               # feature_fraction=0.9
                  "subsample": 0.95,                      # bagging_fraction=0.8
                  "subsample_freq": 1,                    # bagging_freq=1
                  "reg_alpha": 0,
                  "reg_lambda": 0.01,
                  "random_state": 2022,
                  "n_jobs": -1,
                  "verbose": -1}

    # Loading all the data
    # ----------------------------
    print("\n")
    total_df = load_pkl("total_df.pkl")
    feat_df, target_df = load_pkl("lgb_dense_feat.pkl")
    train_df = total_df[total_df["ID"].isnull()].reset_index(drop=True)
    test_df = total_df[total_df["ID"].notnull()].reset_index(drop=True)
    key_cols = ["ID", "QUEUE_ID", "DOTTING_TIME"]


    # Model training
    # ----------------------------
    train_feat_df = feat_df[feat_df["ID"].isnull()].reset_index(drop=True)
    test_feat_df = feat_df[feat_df["ID"].notnull()].reset_index(drop=True)
    test_feat_df = test_feat_df.groupby(["QUEUE_ID", "ID"]).apply(
        lambda x: x.iloc[-1]).reset_index(drop=True)

    oof_pred_df = pd.DataFrame(None)
    test_pred_df = test_feat_df[["ID"]].copy()
    total_scores_df = np.zeros((10, 4))

    target_cols = list(target_df.drop(["QUEUE_ID", "DOTTING_TIME"], axis=1).columns)
    for ind, target_name in enumerate(target_cols):
        target_vals = target_df[target_name].values
        train_feat_vals = train_feat_df.drop(key_cols, axis=1).values
        test_feat_vals = test_feat_df.drop(key_cols, axis=1).values

        train_feat_vals = train_feat_vals[~np.isnan(target_vals)]
        target_vals = target_vals[~np.isnan(target_vals)]

        folds = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED)
        y_pred_tmp = np.zeros((len(test_feat_vals), ))
        oof_pred_tmp = np.zeros((len(train_feat_vals), ))

        print("\n[INFO] Target Name: {}".format(target_name))
        print("[INFO] #training samples: {}, #testing samples: {}, #feats: {}".format(
            len(train_feat_vals), len(test_feat_vals), train_feat_vals.shape[1]))
        print("==================================")
        for fold, (tra_id, val_id) in enumerate(folds.split(train_feat_vals,
                                                            target_vals)):
            d_train, d_valid = train_feat_vals[tra_id], train_feat_vals[val_id]
            t_train, t_valid = target_vals[tra_id], target_vals[val_id]

            d_train = lgb.Dataset(d_train,
                                  label=t_train)
            d_valid = lgb.Dataset(d_valid, 
                                  label=t_valid,
                                  reference=d_train)
            reg = lgb.train(lgb_params,
                            d_train, valid_sets=d_valid,
                            early_stopping_rounds=EARLY_STOP_ROUNDS,
                            verbose_eval=False)

            valid_pred = reg.predict(train_feat_vals[val_id],
                                      num_iteration=reg.best_iteration)
            oof_pred_tmp[val_id] = valid_pred
            y_pred_tmp += reg.predict(test_feat_vals, num_iteration=reg.best_iteration)/N_FOLDS

            valid_mse = mean_squared_error(
                t_valid.reshape((-1, 1)), valid_pred.reshape((-1, 1)))
            valid_mae = mean_absolute_error(
                t_valid.reshape((-1, 1)), valid_pred.reshape((-1, 1)))
            valid_r2 = r2_score(
                t_valid.reshape((-1, 1)), valid_pred.reshape((-1, 1)))

            print("-- fold {}({}): valid MSE: {:.5f}, MAE: {:.5f}, R2: {:.5f}".format(
                fold, N_FOLDS, valid_mse, valid_mae, valid_r2))
            lgb_params["random_state"] += 2099

        # Save current round prediction results
        total_scores_df[ind, 0] = mean_squared_error(target_vals.reshape((-1, 1)),
                                                      oof_pred_tmp.reshape((-1, 1)))
        total_scores_df[ind, 1] = mean_absolute_error(target_vals.reshape((-1, 1)),
                                                      oof_pred_tmp.reshape((-1, 1)))
        total_scores_df[ind, 2] = r2_score(target_vals.reshape((-1, 1)),
                                            oof_pred_tmp.reshape((-1, 1)))
        total_scores_df[ind, 3] = np.mean(custom_metric(target_vals, np.clip(oof_pred_tmp, 
                                                                             a_min=0,
                                                                             a_max=np.max(oof_pred_tmp))))
        test_pred_df[target_name] = y_pred_tmp

        print("-- Total: total valid MSE: {:.5f}, MAE: {:.5f}, R2: {:.5f}".format(
                total_scores_df[ind, 0], total_scores_df[ind, 1], total_scores_df[ind, 2]))
        print("==================================")

    # Preparing the evaluation dataframe
    total_scores = pd.DataFrame(
        total_scores_df, columns=["oof_mse", "oof_mae", "oof_r2", "oof_custom"])
    total_scores["target_name"] = list(test_pred_df.drop(["ID"], axis=1).columns)

    # Preparing submissions
    # ----------------------------
    custom_score = 0
    for i in range(1, 5+1):
        custom_score += total_scores[total_scores["target_name"] == "CPU_USAGE_{}".format(i)]["oof_mae"].values[0] / 100 * 0.9
        custom_score += total_scores[total_scores["target_name"] == "LAUNCHING_JOB_NUMS_{}".format(i)]["oof_custom"].values[0] * 0.1
    custom_score = 1 - custom_score

    sub_ind = len(os.listdir(".//submissions//")) + 1
    sub_name = "{}_lgb_{}_cpu_mae_{:.3f}_jobs_mae_{:.3f}_scores_{:.3f}".format(
        sub_ind,
        N_FOLDS,
        np.mean(total_scores["oof_mae"][total_scores["target_name"].apply(lambda x: "CPU" in x)].values),
        np.mean(total_scores["oof_mae"][total_scores["target_name"].apply(lambda x: "JOB" in x)].values),
        custom_score)

    submission = pd.read_csv(".//data//submit_example.csv")
    submission_to_save = test_pred_df.sort_values(by=["ID"]).copy()
    submission_to_save.reset_index(drop=True, inplace=True)
    submission_to_save = np.round(submission_to_save).astype(int)
    submission_to_save = submission_to_save.clip(0, )

    submission_to_save = submission_to_save[submission.columns]
    submission_to_save.to_csv(".//submissions//{}.csv".format(sub_name), encoding="utf-8", index=False)
