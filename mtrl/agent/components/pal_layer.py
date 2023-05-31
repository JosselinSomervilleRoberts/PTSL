import torch
import torch.nn as nn
from typing import Optional
from toolbox.printing import debug as print_debug


class PALLayer(nn.Module):
    """Implementation of a PAL Layer."""

    def __init__(self, input_size: int, output_size: int, pal_size: int, n_tasks: int, project_down: Optional[nn.Linear] = None, project_up: Optional[nn.Linear] = None, activation: nn.Module = nn.ReLU(), debug: bool = False):
        super(PALLayer, self).__init__()
        self.activation = activation
        self.n_tasks = n_tasks
        self.pal_size = pal_size
        self.input_size = input_size
        self.output_size = output_size
        self._debug = debug

        # Shared Linear, it is a vanilla Linear layer
        self.shared_linear = nn.Linear(input_size, output_size)

        # Individual Linear. This represents n_tasks linear layers of pal_size to pal_size
        self.individual_linears_weight = nn.Parameter(torch.randn(n_tasks, pal_size, pal_size))
        self.individual_linears_bias = nn.Parameter(torch.zeros(n_tasks, pal_size))

        # Project Down
        self.project_down = project_down
        if self.project_down is None:
            self.project_down = nn.Linear(in_features=input_size, out_features=pal_size, bias=False)
        assert self.project_down.in_features == input_size
        assert self.project_down.out_features == pal_size

        # Project Up
        self.project_up = project_up
        if self.project_up is None:
            self.project_up = nn.Linear(in_features=pal_size, out_features=output_size, bias=False)
        assert self.project_up.in_features == pal_size
        assert self.project_up.out_features == output_size

    def set_shared_linear(self, weights: torch.Tensor, bias: torch.Tensor):
        assert weights.shape == (self.output_size, self.input_size), f"Expected shape for shared weight {(self.output_size, self.input_size)}, got {weights.shape}"
        assert bias.shape == (self.output_size,), f"Expected shape for shared bias {(self.output_size,)}, got {bias.shape}"
        self.shared_linear.weight = nn.Parameter(weights)
        self.shared_linear.bias = nn.Parameter(bias)

    def set_individual_linear(self, task_id: int, weights: torch.Tensor, bias: torch.Tensor):
        assert weights.shape == (self.pal_size, self.pal_size), f"Expected shape for individual weight {task_id} {(self.pal_size, self.pal_size)}, got {weights.shape}"
        assert bias.shape == (self.pal_size,), f"Expected shape for individual bias {task_id} {(self.pal_size,)}, got {bias.shape}"
        self.individual_linears_weight.data[task_id] = weights
        self.individual_linears_bias.data[task_id] = bias

    def set_project_down(self, weights: torch.Tensor):
        assert weights.shape == (self.pal_size, self.input_size), f"Expected shape for project down {(self.pal_size, self.input_size)}, got {weights.shape}"
        self.project_down.weight = nn.Parameter(weights)

    def set_project_up(self, weights: torch.Tensor):
        assert weights.shape == (self.output_size, self.pal_size), f"Expected shape for project up {(self.output_size, self.pal_size)}, got {weights.shape}"
        self.project_up.weight = nn.Parameter(weights)

    def set_debug(self, debug: bool):
        self._debug = debug

    def forward(self, x: torch.Tensor):
        assert x.shape == (self.n_tasks, self.input_size), f"Expected shape {(self.n_tasks, self.input_size)}, got {x.shape}"
        if self._debug: print_debug(x)

        # Shared linear
        y_shared = self.shared_linear(x)
        if self._debug: print_debug(y_shared)

        # Individual linears
        # 1. Project down
        x_down = self.project_down(x)
        if self._debug: print_debug(x_down)

        # 2. Compute X @ Wi for all i efficiently using broadcasting
        x_down = x_down.unsqueeze(1)
        y_down = torch.matmul(x_down, self.individual_linears_weight)
        y_down = y_down.squeeze(1) + self.individual_linears_bias
        if self._debug: print_debug(y_down)

        # 3. Project up
        y_individual = self.project_up(y_down)
        if self._debug: print_debug(y_individual)

        # Sum and apply activation
        y = y_shared + y_individual
        y = self.activation(y)
        if self._debug: print_debug(y)
        return y
