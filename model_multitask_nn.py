#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  2 23:46:56 2020

@author: zhuoyin94
"""

import os
import gc
import json
import pandas as pd
import numpy as np
from tqdm import tqdm
import warnings
from datetime import datetime

import tensorflow as tf
import tensorflow.keras
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Input, Dense, Embedding, Dropout, BatchNormalization, PReLU, Add
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, Callback

from numba import njit
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import StratifiedKFold, KFold
import lightgbm as lgb

from utils import LoadSave, custom_metric

np.random.seed(2021)
warnings.filterwarnings('ignore')
sns.set(style="ticks", font_scale=1.2, palette='deep', color_codes=True)
colors = ["red"] + ["C{}".format(i) for i in range(9+1)]

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Restrict TensorFlow to only use the first GPU
        tf.config.experimental.set_visible_devices(gpus[0], 'GPU')

        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)
###############################################################################
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
    """Compute the lagging statistica features based on a specific time_range(lagging_mins)(Maximum == 20)."""
    if feat_col_name is None:
        raise ValueError("The lagging statistical feature name is not specified !")
    if operation_list is None:
        operation_list = [np.mean]
    df_index = np.arange(len(df))
    df_time_index = df[time_col_name].values
    df_feat_vals = df[feat_col_name].values

    # Scan the value according to the window
    feat_vals_to_compute = njit_scan(df_index,
                                     df_time_index,
                                     df_feat_vals,
                                     lagging_mins)

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


def build_model(verbose=False, is_compile=True, **kwargs):
    """
    ----------
    Author: Michael Yin
    E-Mail: zhuoyin94@163.com
    ----------
    @Description:
    ----------
    Multi-level text matching model.
    @Parameters:
    ----------
    verbose: {bool-like}
        If True, print the model information before training started.
    is_complie: {bool-like}
        If True, complie the model in this method.
    **kwargs:
        Other important parameters related to training process.
    @Return:
    ----------
    A constructed model.
    """

    n_feats = kwargs.pop("n_feats", 128)
    layer_input = Input(shape=(n_feats, ),
                        dtype="float32",
                        name="input_feats")

    layer_dense_init = Dense(32, activation="relu")(layer_input)
    layer_dense_norm = BatchNormalization()(layer_dense_init)
    layer_dense_dropout = Dropout(0.3)(layer_dense_norm)
    layer_dense_prelu = PReLU()(layer_dense_dropout)

    layer_dense = Dense(16, activation="relu")(layer_dense_prelu)
    layer_dense_norm = BatchNormalization()(layer_dense)
    layer_dense_dropout = Dropout(0.3)(layer_dense_norm)
    layer_dense_prelu = PReLU()(layer_dense_dropout)

    layer_dense = Dense(32, activation="relu")(layer_dense_prelu)
    layer_dense_norm = BatchNormalization()(layer_dense)
    layer_dense_dropout = Dropout(0.3)(layer_dense_norm)
    layer_dense_prelu = PReLU()(layer_dense_dropout)

    # Residual Connection
    # ----------------------------
    layer_total = Add()([layer_dense_init, layer_dense_prelu])
    layer_pred = Dense(2, activation="relu", name="output")(layer_total)
    model = Model([layer_input], layer_pred)

    if verbose:
        model.summary()
    if is_compile:
        model.compile(loss="mean_absolute_error",
                      optimizer=Adam(0.01))
    return model


if __name__ == "__main__":
    # Pre-setting global parameters:
    # ----------------------------
    N_FOLDS = 2
    N_EPOCHS = 4
    BATCH_SIZE = 32768
    RANDOM_SEED = 2020
    EARLY_STOP_ROUNDS = 50

    # Loading all the data
    # ----------------------------
    print("\n")
    total_df = load_pkl("total_df.pkl")
    feat_df, target_df = load_pkl("nn_dense_feat.pkl")
    train_df = total_df[total_df["ID"].isnull()].reset_index(drop=True)
    test_df = total_df[total_df["ID"].notnull()].reset_index(drop=True)

    key_cols = ["ID", "QUEUE_ID", "DOTTING_TIME"]
    cat_cols = ["STATUS", "QUEUE_TYPE", "PLATFORM", "RESOURCE_TYPE", "CU"]
    numeric_cols = ["CPU_USAGE", "MEM_USAGE", "LAUNCHING_JOB_NUMS",
                    "RUNNING_JOB_NUMS", "SUCCEED_JOB_NUMS", "CANCELLED_JOB_NUMS",
                    "FAILED_JOB_NUMS", "DISK_USAGE"]


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
    for i in range(1, 5+1):
        target_names = ["CPU_USAGE_{}".format(i),
                        "LAUNCHING_JOB_NUMS_{}".format(i)]
        target_vals = target_df[target_names].values

        # Select the valid feature values
        train_feat_vals = train_feat_df.drop(key_cols, axis=1).values
        train_feat_vals = train_feat_vals[~np.isnan(target_vals[:, 0])]

        test_feat_vals = test_feat_df.drop(key_cols, axis=1).values
        target_vals = target_vals[~np.isnan(target_vals[:, 0])]

        # Padding the np.nan values && normalizing data
        for j in range(train_feat_vals.shape[1]):
            tmp_mean_val = np.nanmean(train_feat_vals[:, j])
            train_feat_vals[:, j][np.isnan(train_feat_vals[:, j])] = tmp_mean_val

        for j in range(test_feat_vals.shape[1]):
            tmp_mean_val = np.nanmean(test_feat_vals[:, j])
            test_feat_vals[:, j][np.isnan(test_feat_vals[:, j])] = tmp_mean_val

        X_sc = StandardScaler()
        X_sc.fit(train_feat_vals)
        train_feat_vals = X_sc.transform(train_feat_vals)
        test_feat_vals = X_sc.transform(test_feat_vals)

        # Split folds
        folds = KFold(n_splits=N_FOLDS, shuffle=True)
        y_pred_tmp = np.zeros((len(test_feat_vals), 2))
        oof_pred_tmp = np.zeros((len(train_feat_vals), 2))

        early_stop = EarlyStopping(monitor='val_loss', mode='min',
                                   verbose=1, patience=EARLY_STOP_ROUNDS,
                                   restore_best_weights=True)

        print("\n[INFO] Target Name: {}(Training NN)".format(target_names))
        print("[INFO] #training samples: {}, #testing samples: {}, #feats: {}".format(
            len(train_feat_vals), len(test_feat_vals), train_feat_vals.shape[1]))
        print("==================================")
        for fold, (tra_id, val_id) in enumerate(folds.split(train_feat_vals,
                                                            target_vals)):
            d_train, d_valid = train_feat_vals[tra_id], train_feat_vals[val_id]
            t_train, t_valid = target_vals[tra_id], target_vals[val_id]

            # Destroy all graph nodes in memory
            K.clear_session()
            gc.collect()

            # Building and training model
            model = build_model(verbose=False,
                                is_complie=True,
                                n_feats=d_train.shape[1])

            history = model.fit(x=[d_train], y=t_train,
                                batch_size=BATCH_SIZE,
                                epochs=N_EPOCHS,
                                validation_data=([d_valid], t_valid),
                                callbacks=[early_stop],
                                verbose=0)

            valid_pred = model.predict([d_valid],
                                       use_multiprocessing=True)
            oof_pred_tmp[val_id] = valid_pred
            y_pred_tmp += model.predict([test_feat_vals],
                                        use_multiprocessing=True)/N_FOLDS

            for j in range(2):
                valid_mse = mean_squared_error(
                    t_valid[:, j].reshape((-1, 1)), valid_pred[:, j].reshape((-1, 1)))
                valid_mae = mean_absolute_error(
                    t_valid[:, j].reshape((-1, 1)), valid_pred[:, j].reshape((-1, 1)))
                valid_r2 = r2_score(
                    t_valid[:, j].reshape((-1, 1)), valid_pred[:, j].reshape((-1, 1)))
                print("-- fold {}({}): valid MSE: {:.5f}, MAE: {:.5f}, R2: {:.5f}".format(
                    fold, N_FOLDS, valid_mse, valid_mae, valid_r2))
        print("\n")

        # Save current round prediction results
        for j in range(2):
            total_scores_df[(i-1)*2+j, 0] = mean_squared_error(target_vals[:, j].reshape((-1, 1)),
                                                               oof_pred_tmp[:, j].reshape((-1, 1)))
            total_scores_df[(i-1)*2+j, 1] = mean_absolute_error(target_vals[:, j].reshape((-1, 1)),
                                                                oof_pred_tmp[:, j].reshape((-1, 1)))
            total_scores_df[(i-1)*2+j, 2] = r2_score(target_vals[:, j].reshape((-1, 1)),
                                                     oof_pred_tmp[:, j].reshape((-1, 1)))
            total_scores_df[(i-1)*2+j, 3] = np.mean(custom_metric(
                target_vals[:, j], np.clip(oof_pred_tmp[:, j], a_min=0, a_max=np.max(oof_pred_tmp[:, j]))))
            test_pred_df[target_names[j]] = y_pred_tmp[:, j]
            print("-- {}: total valid MSE: {:.5f}, MAE: {:.5f}, R2: {:.5f}".format(
                target_names[j],
                total_scores_df[(i-1)*2+j, 0],
                total_scores_df[(i-1)*2+j, 1],
                total_scores_df[(i-1)*2+j, 2]))
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
    sub_name = "{}_nn_{}_cpu_mae_{:.3f}_jobs_mae_{:.3f}_scores_{:.3f}".format(
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
