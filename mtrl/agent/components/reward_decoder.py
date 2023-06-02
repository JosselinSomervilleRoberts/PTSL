# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# Implementation based on Denis Yarats' implementation of [SAC](https://github.com/denisyarats/pytorch_sac).
"""Reward decoder component for the agent."""

from typing import List
from toolbox.printing import str_with_color

import torch.nn as nn

from mtrl.agent.components import base as base_component
from mtrl.agent.components import moe_layer
from mtrl.utils.types import ModelType, TensorType


class RewardDecoder(base_component.Component):
    def __init__(
        self,
        feature_dim: int,
    ):
        """Predict reward using the observations.

        Args:
            feature_dim (int): dimension of the feature used to predict
                the reward.
        """
        super().__init__()
        self.trunk = moe_layer.SequentialSum(
            nn.Linear(feature_dim, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Linear(512, 1),
        )

    def summary(self, prefix: str = "") -> str:
        """Summary of the RewardDecoder.

        Args:
            prefix (str, optional): prefix to add to the summary before each line.
                Defaults to "".

        Returns:
            str: summary of the RewardDecoder.
        """
        summary: str = ""
        num_parameters = sum(p.numel() for p in self.parameters() if p.requires_grad)
        summary += f"{prefix}Reward Decoder " + str_with_color(f"({num_parameters} parameters)", "purple") + "\n"
        summary += f"{prefix}Trunk:\n"
        summary += self.trunk.summary(prefix=f"{prefix}    ")
        return summary
    
    def __repr__(self) -> str:
        return self.summary()

    def forward(self, x: TensorType) -> TensorType:
        return self.trunk(x)

    def get_last_shared_layers(self) -> List[ModelType]:
        return [self.trunk[-1]]
