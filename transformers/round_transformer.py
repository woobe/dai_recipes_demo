## Original: https://github.com/h2oai/driverlessai-recipes/blob/master/transformers/numeric/round_transformer.py

"""Rounds numbers to 1, 2 or 3 decimals"""
from h2oaicore.transformer_utils import CustomTransformer
import datatable as dt
import numpy as np


class JoeSupaDupaRoundTransformer(CustomTransformer):
    @staticmethod
    def get_parameter_choices():
        return {"decimals": [1, 2, 3]}

    @property
    def display_name(self):
        return "JoeSupaDupaRound%dDecimals" % self.decimals

    def __init__(self, decimals, **kwargs):
        super().__init__(**kwargs)
        self.decimals = decimals

    def fit_transform(self, X: dt.Frame, y: np.array = None):
        return self.transform(X)

    def transform(self, X: dt.Frame):
        return np.round(X.to_numpy(), decimals=self.decimals)
