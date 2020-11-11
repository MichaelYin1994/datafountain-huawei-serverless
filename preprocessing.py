#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 18 16:37:27 2020

@author: zhuoyin94
"""

import numpy as np
import pandas as pd
from utils import LoadSave
import warnings

np.random.seed(2020)
warnings.filterwarnings('ignore')
###############################################################################
def load_csv(file_name=None, path_name=".//cached_data//", nrows=100):
    """Load the original *.csv data."""
    total_name = path_name + file_name
    csv_data = pd.read_csv(total_name, nrows=nrows)
    return csv_data


if __name__ == "__main__":
    train_df = load_csv(file_name="train.csv",
                        path_name=".//data//",
                        nrows=None)
    test_df = load_csv(path_name=".//data//",
                       file_name="evaluation_public.csv",
                       nrows=None)
    total_df = pd.concat([train_df, test_df], axis=0)

    # Encoding category variables
    # --------------------------------
    cat_list = ["STATUS", "QUEUE_TYPE", "PLATFORM", "RESOURCE_TYPE"]
    for name in cat_list:
        total_df[name] = total_df[name].astype("category").cat.codes
    total_df.sort_values(by=["QUEUE_ID", "DOTTING_TIME"], ascending=True, inplace=True)
    total_df.reset_index(drop=True, inplace=True)

    # Save data to local path
    # --------------------------------
    file_processor = LoadSave()
    file_processor.save_data(path=".//cached_data//total_df.pkl",
                              data=total_df)
