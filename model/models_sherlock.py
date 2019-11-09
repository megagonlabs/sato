from typing import List, Dict, Any
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F


class FeatureEncoder(nn.Module):
    def __init__(self,
                 name: str,
                 input_dim: int,
                 embedding_dim: int = 300,
                 dropout_ratio: float = 0.5,
                 skip_conversion: bool = False):
        super(FeatureEncoder, self).__init__()
        self.name = name
        self.input_dim = input_dim
        self.embedding_dim = embedding_dim
        self.skip_conversion = skip_conversion
        if self.skip_conversion:
            self.embedding_dim = self.input_dim

        # TODO(Yoshi): Check if applying Batch normalization to the input is good
        self.bn1 = nn.BatchNorm1d(num_features=input_dim)
        self.linear1 = nn.Linear(input_dim,
                                 embedding_dim)
        self.relu1 = nn.ReLU()
        self.dp1 = nn.Dropout(dropout_ratio)
        self.linear2 = nn.Linear(embedding_dim,
                                 embedding_dim)
        self.relu2 = nn.ReLU()

    def forward(self,
                x: torch.Tensor):
        out = self.bn1(x)
        if not self.skip_conversion:
            out = self.relu1(self.linear1(out))
            out = self.dp1(out)
            out = self.linear2(out)
            out = self.relu2(out)
        return out


class RestEncoder(nn.Module):
    """
    RestEncoder does not use Fully-connected layers
    Will merge into FeatureEncoder
    """
    def __init__(self,
                 input_dim: int,
                 embedding_dim: int = 300):
        super(RestEncoder, self).__init__()
        self.input_dim = input_dim
        self.bn1 = nn.BatchNorm1d(num_features=input_dim)

    def forward(self, x):
        out = self.bn1(x)
        return out


class SingleFeatureClassifier(nn.Module):
    def __init__(self,
                 encoder: nn.Module,
                 input_dim: int,
                 num_classes: int = 78):
        super(SingleFeatureClassifier, self).__init__()
        self.linear(input_dim,
                    num_classes)

    def forward(self,
                x: torch.Tensor):
        out = self.encoder(x)
        # TODO(Yoshi): Probably need ReLU&dropout here.
        out = self.linear(x)
        out = F.softmax(out)

        return out


class SherlockClassifier(nn.Module):
    def __init__(self,
                 encoders: Dict[str, nn.Module],
                 embedding_dim: int = 500,
                 num_classes: int = 78,
                 dropout_ratio: float = 0.5): # params: Dict[str, Any]):
        super(SherlockClassifier, self).__init__()
        self.encoders = encoders
        # Register encoders as parameters
        for n, e in self.encoders.items():
            self.add_module("FeatureEncoder_{}".format(n), e)

        self.feature_names = sorted(encoders.keys()) # Fix the order of encoders
        # Sum of input_dim of all encoders
        total_input_dim = sum([x.embedding_dim for x in encoders.values()])
        self.bn1 = nn.BatchNorm1d(num_features=total_input_dim)
        self.linear1 = nn.Linear(total_input_dim, embedding_dim)
        self.relu1 = nn.ReLU()
        self.dp1 = nn.Dropout(dropout_ratio)
        self.linear2 = nn.Linear(embedding_dim, embedding_dim)
        self.relu2 = nn.ReLU()
        self.linear3 = nn.Linear(embedding_dim, num_classes)
        self.relu2 = nn.ReLU()


    def _concat(self,
                X: Dict[str, torch.Tensor]):
        embs = []
        for name in self.feature_names:
            x = X[name]
            emb = self.encoders[name](x)
            embs.append(emb)

        concat = torch.cat(embs, 1)
        return concat

    def embedding(self,
                  X: Dict[str, torch.Tensor]):
        concat = self._concat(X)
        out = self.bn1(concat)
        out = self.linear1(out)
        out = self.relu1(out)
        out = self.dp1(out)
        out = self.linear2(out)
        out = self.relu2(out)
        return out

    def forward(self,
                X: Dict[str, torch.Tensor]):
        out = self.embedding(X)
        # TODO(Yoshi): Probably, need relu&droput?
        out = self.linear3(out)
        return out

    def predict(self,
                X: Dict[str, torch.Tensor]):
        return F.softmax(self.forward(X))




input_dim_dict = {"char": 960,
                          "word": 200,
                          "par": 400,
                          "rest": 27} 
embedding_dim_dict = {"char": 300,
                          "word": 200,
                          "par": 400,
                          "rest": 27} # Dummy. Does not matter as long as skip_conversion is ON

def build_sherlock(arg_feature_groups, num_classes, topic_dim=None, dropout_ratio=0.5):
    # convinient function to build a sherlock instance 
    feature_enc_dict = {}

    for feature_name in arg_feature_groups:
        if feature_name in ["rest"]:
            skip_conversion = True
        else:
            skip_conversion = False
        feature_enc = FeatureEncoder(feature_name,
                                     input_dim=input_dim_dict[feature_name],
                                     embedding_dim=embedding_dim_dict[feature_name],
                                     skip_conversion=skip_conversion,
                                     dropout_ratio=dropout_ratio)
        feature_enc_dict[feature_name] = feature_enc

    if topic_dim is not None:
        topic_dim = int(topic_dim)

        feature_enc_dict['topic'] = FeatureEncoder('topic',
                                     input_dim=topic_dim,
                                     embedding_dim=topic_dim,
                                     skip_conversion=False,
                                     dropout_ratio=dropout_ratio)

    assert len(feature_enc_dict) > 0
    classifier = SherlockClassifier(feature_enc_dict, num_classes=num_classes, dropout_ratio=dropout_ratio)
    return classifier


