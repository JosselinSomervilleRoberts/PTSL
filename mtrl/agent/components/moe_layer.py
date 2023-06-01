# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""Layers for parallelizing computation with mixture of experts.

A mixture of experts(models) can be easily simulated by maintaining a list
of models and iterating over them. However, this can be slow in practice.
We provide some additional modules which makes it easier to create mixture
of experts without slowing down training/inference.
"""
from typing import Dict, List, Optional

import torch
from torch import nn

from mtrl.agent import utils as agent_utils
from mtrl.agent.ds.task_info import TaskInfo
from mtrl.utils.types import ConfigType, TensorType
from mtrl.agent.components.pal_layer import PALLayer
from toolbox.printing import debug as print_debug
from wandb.vendor.pynvml.pynvml import NVML_THERMAL_CONTROLLER_NVSYSCON_CANOAS



class Sequential(nn.Sequential):
    """Functions exactly as torch.nn.Sequential, but has a summary method."""

    def summary(self, prefix: str = "") -> str:
        """Summary of the Sequential.

        Args:
            prefix (str, optional): prefix to add to the summary before each line.
                Defaults to "".

        Returns:
            str: summary of the Sequential.
        """
        summary: str = ""
        num_parameters = sum(p.numel() for p in self.parameters() if p.requires_grad)
        summary += f"{prefix}Sequential ({num_parameters} parameters)\n"
        for i, layer in enumerate(self):
            # If the layer has a summary method, use it.
            if hasattr(layer, "summary"):
                summary += f"{prefix}\tLayer {i}:\n"
                summary += layer.summary(prefix=prefix + "\t\t")
            else:
                summary += f"{prefix}\tLayer {i}: {layer}\n"
        return summary
    
    def __repr__(self) -> str:
        return self.summary()

class ModuleList(nn.ModuleList):
    """Functions exactly as torch.nn.ModuleList, but has a summary method."""

    def summary(self, prefix: str = "") -> str:
        """Summary of the ModuleList.

        Args:
            prefix (str, optional): prefix to add to the summary before each line.
                Defaults to "".

        Returns:
            str: summary of the ModuleList.
        """
        summary: str = ""
        num_parameters = sum(p.numel() for p in self.parameters() if p.requires_grad)
        summary += f"{prefix}ModuleList ({num_parameters} parameters)\n"
        for i, layer in enumerate(self):
            # If the layer has a summary method, use it.
            if hasattr(layer, "summary"):
                summary += f"{prefix}\tLayer {i}:\n"
                summary += layer.summary(prefix=prefix + "\t\t")
            else:
                summary += f"{prefix}\tLayer {i}: {layer}\n"
        return summary
    
    def __repr__(self) -> str:
        return self.summary()

class Linear(nn.Module):
    def __init__(
        self, num_experts: int, in_features: int, out_features: int, bias: bool = True
    ):
        """torch.nn.Linear layer extended for use as a mixture of experts.

        Args:
            num_experts (int): number of experts in the mixture.
            in_features (int): size of each input sample for one expert.
            out_features (int): size of each output sample for one expert.
            bias (bool, optional): if set to ``False``, the layer will
                not learn an additive bias. Defaults to True.
        """
        super().__init__()
        self.num_experts = num_experts
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(
            torch.rand(self.num_experts, self.in_features, self.out_features)
        )
        if bias:
            self.bias = nn.Parameter(torch.rand(self.num_experts, 1, self.out_features))
            self.use_bias = True
        else:
            self.use_bias = False

    def summary(self, prefix: str = "") -> str:
        """Summary of the Linear.

        Args:
            prefix (str, optional): prefix to add to the summary before each line.
                Defaults to "".

        Returns:
            str: summary of the Linear.
        """
        summary: str = ""
        num_parameters = sum(p.numel() for p in self.parameters() if p.requires_grad)
        summary += f"{prefix}Linear ({num_parameters} parameters)\n"
        summary += f"{prefix}\tWeight: Tensor ({self.num_experts}, {self.in_features}, {self.out_features})\n"
        if self.use_bias:
            summary += f"{prefix}\tBias: Tensor ({self.num_experts}, 1, {self.out_features})\n"
        else:
            summary += f"{prefix}\tBias: None\n"
        return summary
    
    def __repr__(self) -> str:
        return self.summary()

    def forward(self, x: TensorType) -> TensorType:
        if self.use_bias:
            return x.matmul(self.weight) + self.bias
        else:
            return x.matmul(self.weight)

    def extra_repr(self) -> str:
        return f"num_experts={self.num_experts}, in_features={self.in_features}, out_features={self.out_features}, bias={self.use_bias}"


class FeedForward(nn.Module):
    def __init__(
        self,
        num_experts: int,
        in_features: int,
        out_features: int,
        num_layers: int,
        hidden_features: int,
        bias: bool = True,
    ):
        """A feedforward model of mixture of experts layers.

        Args:
            num_experts (int): number of experts in the mixture.
            in_features (int): size of each input sample for one expert.
            out_features (int): size of each output sample for one expert.
            num_layers (int): number of layers in the feedforward network.
            hidden_features (int): dimensionality of hidden layer in the
                feedforward network.
            bias (bool, optional): if set to ``False``, the layer will
                not learn an additive bias. Defaults to True.
        """
        super().__init__()
        self._layers: List[nn.Module] = []
        current_in_features = in_features
        for _ in range(num_layers - 1):
            linear = Linear(
                num_experts=num_experts,
                in_features=current_in_features,
                out_features=hidden_features,
                bias=bias,
            )
            self._layers.append(linear)
            self._layers.append(nn.ReLU())
            current_in_features = hidden_features
        linear = Linear(
            num_experts=num_experts,
            in_features=current_in_features,
            out_features=out_features,
            bias=bias,
        )
        self._layers.append(linear)
        self._model = Sequential(*self._layers)

    def summary(self, prefix: str = "") -> str:
        """Summary of the FeedForward.

        Args:
            prefix (str, optional): prefix to add to the summary before each line.
                Defaults to "".

        Returns:
            str: summary of the FeedForward.
        """
        summary: str = ""
        num_parameters = sum(p.numel() for p in self.parameters() if p.requires_grad)
        summary += f"{prefix}FeedForward ({num_parameters} parameters)\n"
        summary += f"Layers:\n"
        for i, layer in enumerate(self._layers):
            summary += f"{prefix}  Layer {i}:\n"
            summary += layer.summary(prefix=prefix + "\t")
        return summary
    
    def __repr__(self) -> str:
        return self.summary()

    def forward(self, x: TensorType) -> TensorType:
        return self._model(x)

    def __repr__(self) -> str:
        return str(self._model)
    

class FeedForwardPAL(nn.Module):

    @staticmethod
    def compute_number_of_parameters(
        n_tasks: int,
        in_features: int,
        out_features: int,
        num_layers: int,
        hidden_features: int,
        pal_features: int,
        shared_projection: bool = True,
    ) -> int:
        """
        Computes the number of parameters of a FeedForwardPAL model without actually creating it.
        """
        num_parameters = 0
        num_parameters += PALLayer.compute_number_of_parameters(in_features, hidden_features, pal_features, n_tasks, project_down=True, project_up=True, must_project_up=shared_projection)
        for _ in range(1, num_layers - 1):
            num_parameters += PALLayer.compute_number_of_parameters(hidden_features, hidden_features, pal_features, n_tasks, project_down=not shared_projection, project_up=not shared_projection, must_project_down=shared_projection, must_project_up=shared_projection)
        num_parameters += PALLayer.compute_number_of_parameters(hidden_features, out_features, pal_features, n_tasks, project_down=True, project_up=True, must_project_down=shared_projection)
        return num_parameters

    def __init__(
        self,
        n_tasks: int,
        in_features: int,
        out_features: int,
        num_layers: int,
        hidden_features: int,
        pal_features: int,
        shared_projection: bool = True,
        use_residual_connections: bool = False,
        activation: nn.Module = nn.ReLU(),
        debug: bool = False,
    ):
        """
        A feedforward model of mixture of experts layers and projected attention layers.
        For a given input, the output will go throught several linear layers (shared for each
        tasks, i.e. simple MDP). However, at each step, the feature will be projected down to
        a lower dimension (pal_features), go through a linear layer to produce pal_features
        features, and then projected back to the original dimension. The result is then added
        to the original output of the linear layer. This is repeated for each layer (the projection
        matrices can be shared or not).
        This is a variant of the PAL model from https://arxiv.org/pdf/1902.02671.pdf
        """
        super().__init__()
        self._n_tasks = n_tasks
        self._shared_projection = shared_projection
        self._use_residual_connections = use_residual_connections
        self._debug = debug

        self._project_up_module, self._project_down_module = None, None
        if self._shared_projection:
            self._project_up_module = PALLayer.get_project_up_module(output_size=hidden_features, pal_size=pal_features)
            self._project_down_module = PALLayer.get_project_down_module(input_size=hidden_features, pal_size=pal_features)
        
        self._layers: ModuleList[PALLayer] = ModuleList()
        self._layers.append(PALLayer(input_size=in_features, output_size=hidden_features, pal_size=pal_features, n_tasks=n_tasks, project_down_module=None, project_up_module=self._project_up_module, activation=activation))
        for _ in range(1, num_layers - 1):
            pal_layer = PALLayer(input_size=hidden_features, output_size=hidden_features, pal_size=pal_features, n_tasks=n_tasks, project_down_module=self._project_down_module, project_up_module=self._project_up_module, activation=activation)
            self._layers.append(pal_layer)
        self._layers.append(PALLayer(input_size=hidden_features, output_size=out_features, pal_size=pal_features, n_tasks=n_tasks, project_down_module=self._project_down_module, project_up_module=None, activation=nn.Identity()))

    def summary(self, prefix: str = "") -> str:
        """Summary of the FeedForward PAL.

        Args:
            prefix (str, optional): prefix to add to the summary before each line.
                Defaults to "".

        Returns:
            str: summary of the FeedForward PAL.
        """
        summary: str = ""
        num_parameters = sum(p.numel() for p in self.parameters() if p.requires_grad)
        summary += f"{prefix}FeedForwardPAL ({num_parameters} parameters)\n"
        summary += f"Shared projection: {self._shared_projection}\n"
        summary += f"Use residual connections: {self._use_residual_connections}\n"
        summary += f"Layers:\n"
        for i, layer in enumerate(self._layers):
            summary += f"{prefix}  Layer {i}:\n"
            summary += layer.summary(prefix=prefix + "\t")
        return summary
    
    def __repr__(self) -> str:
        return self.summary()

    def forward(self, x: TensorType) -> TensorType:
        # Here we simply apply the PAL layers one after the other
        # The only twist it if we use_residual_connections, in which case
        # we add the downsampled output to the downsampled input of the next layer.

        if not self._use_residual_connections:
            for layer in self._layers:
                x = layer(x)
                if self._debug: print_debug(x)
            return x
        
        # If we use residual connections, we need to keep track of the downsampled
        # outputs
        x_down = None
        for layer in self._layers:
            x, x_down = layer.residual_forward(x, x_down)
            if self._debug:
                print_debug(x)
                print_debug(x_down)
        return x

    def set_indices(self, indices: torch.Tensor) -> None:
        for layer in self._layers:
            layer.set_indices(indices)


class MaskCache:
    def __init__(
        self,
        num_tasks: int,
        num_eval_episodes: int,
        batch_size: int,
        task_index_to_mask: TensorType,
    ):
        """In multitask learning, using a mixture of models, different tasks
            can be mapped to different combination of models. This utility
            class caches these mappings so that they do not have to be revaluated.

            For example, when the model is training over 10 tasks, and the
            tasks are always ordered, the mapping of task index to encoder indices
            will be the same and need not be recomputed. We take a very simple
            approach here: cache using the number of tasks, since in our case,
            the task ordering during training and evaluation does not change.
            In more complex cases, a mode (train/eval..) based key could be used.

            This gets a little trickier during evaluation. We assume that we are
            running multiple evaluation episodes (per task) at once. So during
            evaluation, the agent is inferring over num_tasks*num_eval_episodes
            at once.

            We have to be careful about not caching the mapping during update because
            neither the task distribution, nor the task ordering, is pre-determined
            during update. So we explicitly exclude the `batch_size` from the list
            of keys being cached.

        Args:
            num_tasks (int): number of tasks.
            num_eval_episodes (int): number of episodes run during evaluation.
            batch_size (int): batch size for update.
            task_index_to_mask (TensorType): mapping of task index to mask.
        """
        self.masks: Dict[int, TensorType] = {}
        self.task_index_to_mask = task_index_to_mask
        keys_to_cache = [num_tasks, num_tasks * num_eval_episodes]
        self.keys_to_cache = {key for key in keys_to_cache if key != batch_size}
        # This is a hack to get some speed up, by reusing the mask

    def summary(self, prefix: str = "") -> str:
        """Summary of the MaskCache.

        Args:
            prefix (str, optional): prefix to add to the summary before each line.
                Defaults to "".

        Returns:
            str: summary of the MaskCache.
        """
        summary: str = ""
        summary = f"{prefix}Mask cache\n"
        return summary
    
    def __repr__(self) -> str:
        return self.summary()

    def get_mask(self, task_info: TaskInfo) -> TensorType:
        """Get the mask corresponding to a given task info.

        Args:
            task_info (TaskInfo):

        Returns:
            TensorType: encoder mask.
        """
        key = len(task_info.env_index)
        if key in self.keys_to_cache:
            if key not in self.masks:
                self.masks[key] = self._make_mask(task_info)
            return self.masks[key]
        return self._make_mask(task_info)

    def _make_mask(self, task_info: TaskInfo):
        env_index = task_info.env_index
        encoder_mask = self.task_index_to_mask[env_index.squeeze()]
        if len(encoder_mask.shape) == 1:
            encoder_mask = encoder_mask.unsqueeze(0)
        return encoder_mask.t().unsqueeze(2).to(env_index.device)


class MixtureOfExperts(nn.Module):
    def __init__(
        self,
        multitask_cfg: ConfigType,
    ):
        """Class for interfacing with a mixture of experts.

        Args:
            multitask_cfg (ConfigType): config for multitask training.
        """
        super().__init__()
        self.multitask_cfg = multitask_cfg
        self.mask_cache: MaskCache

    def summary(self, prefix: str = "") -> str:
        """Summary of the MixtureOfExperts.

        Args:
            prefix (str, optional): prefix to add to the summary before each line.
                Defaults to "".

        Returns:
            str: summary of the MixtureOfExperts.
        """
        summary: str = ""
        summary = f"{prefix}Mixture of experts\n"
        return summary
    
    def __repr__(self) -> str:
        return self.summary()

    def forward(self, task_info: TaskInfo) -> TensorType:
        return self.mask_cache.get_mask(task_info=task_info)


class AttentionBasedExperts(MixtureOfExperts):
    def __init__(
        self,
        num_tasks: int,
        num_experts: int,
        embedding_dim: int,
        hidden_dim: int,
        num_layers: int,
        temperature: bool,
        should_use_soft_attention: bool,
        task_encoder_cfg: ConfigType,
        multitask_cfg: ConfigType,
        topk: Optional[int] = None,
    ):
        super().__init__(multitask_cfg=multitask_cfg)
        self.should_use_soft_attention = should_use_soft_attention
        self.temperature = temperature
        self.should_use_task_encoding = task_encoder_cfg.should_use_task_encoding
        self.should_detach_task_encoding = (
            self.should_use_task_encoding
            and task_encoder_cfg.should_detach_task_encoding
        )
        if not self.should_use_task_encoding:
            self.emb = nn.Embedding(
                num_embeddings=num_tasks,
                embedding_dim=embedding_dim,
            )
            self.emb.apply(agent_utils.weight_init)
        else:
            embedding_dim = multitask_cfg.task_encoder_cfg.model_cfg.embedding_dim
        self.trunk = agent_utils.build_mlp(
            input_dim=embedding_dim,
            hidden_dim=hidden_dim,
            output_dim=num_experts,
            num_layers=num_layers,
        )
        self.trunk.apply(agent_utils.weight_init)
        self.topk = topk
        self._softmax = torch.nn.Softmax(dim=1)

    def summary(self, prefix: str = "") -> str:
        """Summary of the AttentionBasedExperts.

        Args:
            prefix (str, optional): prefix to add to the summary before each line.
                Defaults to "".

        Returns:
            str: summary of the AttentionBasedExperts.
        """
        summary: str = ""
        num_parameters = sum(p.numel() for p in self.parameters() if p.requires_grad)
        summary += f"{prefix}AttentionBasedExperts ({num_parameters} parameters)\n"
        summary += f"{prefix}Trunk:\n"
        summary += self.trunk.summary(prefix=prefix + "\t")
        summary += "\n"
        return summary
    
    def __repr__(self) -> str:
        return self.summary()

    def forward(self, task_info: TaskInfo) -> TensorType:
        if self.should_use_task_encoding:
            emb = task_info.encoding
            if self.should_detach_task_encoding:
                emb = emb.detach()  # type: ignore[union-attr]
        else:
            env_index = task_info.env_index
            if len(env_index.shape) == 2:
                env_index = env_index.squeeze(1)
            emb = self.emb(env_index)
        output = self.trunk(emb)
        gate = self._softmax(output / self.temperature)
        if not self.should_use_soft_attention:
            topk_attention = gate.topk(self.topk, dim=1)
            topk_attention_indices = topk_attention[1]
            hard_attention_mask = torch.zeros_like(gate).scatter_(
                dim=1, index=topk_attention_indices, value=1.0
            )
            gate = gate * hard_attention_mask
            gate = gate / gate.sum(dim=1).unsqueeze(1)
        if len(gate.shape) > 2:
            breakpoint()
        return gate.t().unsqueeze(2)


class ClusterOfExperts(MixtureOfExperts):
    def __init__(
        self,
        num_tasks: int,
        num_experts: int,
        num_eval_episodes: int,
        batch_size: int,
        multitask_cfg: ConfigType,
        env_name: str,
        task_description: Dict[str, str],
        ordered_task_list: List[str],
        mapping_cfg: ConfigType,
    ):
        """Map the ith task to a subset (cluster) of experts.

        Args:
            num_tasks (int): number of tasks.
            num_experts (int): number of experts in the mixture of experts.
            num_eval_episodes (int): number of episodes run during evaluation.
            batch_size (int): batch size for update.
            multitask_cfg (ConfigType): config for multitask training.
            env_name (str): name of the environment. This is used with the
                mapping configuration.
            task_description (Dict[str, str]): dictionary mapping task
                names to descriptions.
            ordered_task_list (List[str]): ordered list of tasks. This is
                needed because the task description is not always ordered.
            mapping_cfg (ConfigType): config for mapping the tasks to subset
                of experts.
        """

        super().__init__(multitask_cfg=multitask_cfg)
        assert env_name in ["metaworld-mt10", "metaworld-mt50"]
        env_name = env_name.split("-")[1]
        clusters = mapping_cfg[env_name]["cluster"]
        task_index_to_encoder_index = torch.zeros(
            (len(task_description), min(num_experts, len(clusters)))
        )
        for task_index, task_name in enumerate(ordered_task_list):
            description = task_description[task_name]
            for encoder_index, cluster_values in enumerate(clusters.values()):
                for word in cluster_values:
                    if word in description:
                        task_index_to_encoder_index[task_index][encoder_index] = 1.0
        self.mask_cache = MaskCache(
            num_tasks=num_tasks,
            num_eval_episodes=num_eval_episodes,
            batch_size=batch_size,
            task_index_to_mask=task_index_to_encoder_index,
        )

    def summary(self, prefix: str = "") -> str:
        """Summary of the ClusterOfExperts.

        Args:
            prefix (str, optional): prefix to add to the summary before each line.
                Defaults to "".

        Returns:
            str: summary of the ClusterOfExperts.
        """
        summary: str = ""
        summary = f"{prefix}Cluster of experts\n"
        return summary
    
    def __repr__(self) -> str:
        return self.summary()


class OneToOneExperts(MixtureOfExperts):
    def __init__(
        self,
        num_tasks: int,
        num_experts: int,
        num_eval_episodes: int,
        batch_size: int,
        multitask_cfg: ConfigType,
    ):
        """Map the output of ith expert with the ith task.

        Args:
            num_tasks (int): number of tasks.
            num_experts (int): number of experts in the mixture of experts.
            num_eval_episodes (int): number of episodes run during evaluation.
            batch_size (int): batch size for update.
            multitask_cfg (ConfigType): config for multitask training.
        """
        super().__init__(multitask_cfg=multitask_cfg)
        assert num_tasks == num_experts
        self.mask_cache = MaskCache(
            num_tasks=num_tasks,
            num_eval_episodes=num_eval_episodes,
            batch_size=batch_size,
            task_index_to_mask=torch.eye(num_tasks),
        )

    def summary(self, prefix: str = "") -> str:
        """Summary of the OneToOneExperts.

        Args:
            prefix (str, optional): prefix to add to the summary before each line.
                Defaults to "".

        Returns:
            str: summary of the OneToOneExperts.
        """
        summary: str = ""
        summary = f"{prefix}One to one mapping of experts\n"
        return summary
    
    def __repr__(self) -> str:
        return self.summary()


class EnsembleOfExperts(MixtureOfExperts):
    def __init__(
        self,
        num_tasks: int,
        num_experts: int,
        num_eval_episodes: int,
        batch_size: int,
        multitask_cfg: ConfigType,
    ):
        """Ensemble of all the experts.

        Args:
            num_tasks (int): number of tasks.
            num_experts (int): number of experts in the mixture of experts.
            num_eval_episodes (int): number of episodes run during evaluation.
            batch_size (int): batch size for update.
            multitask_cfg (ConfigType): config for multitask training.
        """
        super().__init__(multitask_cfg=multitask_cfg)

        self.mask_cache = MaskCache(
            num_tasks=num_tasks,
            num_eval_episodes=num_eval_episodes,
            batch_size=batch_size,
            task_index_to_mask=torch.ones(num_tasks, num_experts),
        )

    def summary(self, prefix: str = "") -> str:
        """Summary of the EnsembleOfExperts.

        Args:
            prefix (str, optional): prefix to add to the summary before each line.
                Defaults to "".

        Returns:
            str: summary of the EnsembleOfExperts.
        """
        summary: str = ""
        summary = f"{prefix}Ensemble of experts\n"
        return summary
    
    def __repr__(self) -> str:
        return self.summary()
