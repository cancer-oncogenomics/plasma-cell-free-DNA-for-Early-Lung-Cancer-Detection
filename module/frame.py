from functools import reduce
import random
import typing as t

import h2o
import pandas as pd
from pandas.api.types import CategoricalDtype


class GsFrame(object):
    """ Generate DataFrame for model training and prediction """

    def __init__(self, dataset_list=None, feature_list=None, axis=0):

        self.axis = axis
        self.dataset = self._dataset(dataset_list)
        self.feature = self._feature(feature_list, axis=axis)
        self.data = self._data()

    @property
    def c_dataset(self):
        return list(self.dataset.columns)

    @property
    def c_features(self):

        return [c for c in self.feature.columns if c != "SampleID"]

    @property
    def samples(self):

        return list(self.data["SampleID"])

    @property
    def as_pd(self):
        return self.data.copy()

    @property
    def as_h2o(self):
        col_types = {c: "float" for c in self.c_features}
        data = h2o.H2OFrame(self.data.copy(), column_types=col_types)
        return data

    @staticmethod
    def _dataset(dataset_list):

        if dataset_list:
            df_dataset = pd.concat([pd.read_csv(f, sep="\t", low_memory=False) for f in dataset_list], ignore_index=True, sort=False)
        else:
            df_dataset = pd.DataFrame(columns=["SampleID", "Response"])
        return df_dataset

    @staticmethod
    def _feature(feature_list, axis):
        if feature_list and axis == 0:
            df_feature = pd.concat([pd.read_csv(f, low_memory=False) for f in feature_list], ignore_index=True, sort=False)
        elif feature_list and axis == 1:
            df_feature = reduce(lambda x, y: pd.merge(x, y, on="SampleID", how="outer"), [pd.read_csv(f) for f in feature_list])
        else:
            df_feature = pd.DataFrame(columns=["SampleID"])
        return df_feature

    def _data(self):
        """combine info and feature"""

        if len(self.feature) and len(self.dataset):
            df_final = pd.merge(self.feature, self.dataset, on="SampleID", how="inner")
        elif len(self.feature):
            df_final = self.feature.copy()
        elif len(self.dataset):
            df_final = self.dataset.copy()
        else:
            df_final = pd.DataFrame()

        return df_final