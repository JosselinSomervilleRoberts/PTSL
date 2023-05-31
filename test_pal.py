import torch
import torch.nn as nn
from typing import Optional
from toolbox.printing import debug as print_debug


class PAL_Layer(nn.Module):
    def __init__(self, input_size: int, output_size: int, pal_size: int, n_tasks: int, project_down: Optional[nn.Linear] = None, project_up: Optional[nn.Linear] = None, activation: nn.Module = nn.ReLU()):
        super(PAL_Layer, self).__init__()
        self.activation = activation
        self.n_tasks = n_tasks
        self.pal_size = pal_size
        self.input_size = input_size
        self.output_size = output_size

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

    def forward(self, x: torch.Tensor, debug: bool):
        assert x.shape == (self.n_tasks, self.input_size), f"Expected shape {(self.n_tasks, self.input_size)}, got {x.shape}"
        if debug: print_debug(x)

        # Shared linear
        y_shared = self.shared_linear(x)
        if debug: print_debug(y_shared)

        # Individual linears
        # 1. Project down
        x_down = self.project_down(x)
        if debug: print_debug(x_down)

        # 2. Compute X @ Wi for all i efficiently using broadcasting
        x_down = x_down.unsqueeze(1)
        y_down = torch.matmul(x_down, self.individual_linears_weight)
        y_down = y_down.squeeze(1) + self.individual_linears_bias
        if debug: print_debug(y_down)

        # 3. Project up
        y_individual = self.project_up(y_down)
        if debug: print_debug(y_individual)

        # Sum and apply activation
        y = y_shared + y_individual
        y = self.activation(y)
        if debug: print_debug(y)
        return y


def test_pal_layer():
    """Test a PAL Layer to make sure ut works as expected."""
    in_dim, out_dim, pal_dim, n_tasks = 3, 4, 2, 5

    # Creates individually all the components of the PAL Layer
    shared_linear = nn.Linear(in_dim, out_dim)
    individual_linears: nn.ModuleList[nn.Linear] = nn.ModuleList()
    for _ in range(n_tasks):
        individual_linears.append(nn.Linear(pal_dim, pal_dim))
    project_down = nn.Linear(in_dim, pal_dim, bias=False)
    project_up = nn.Linear(pal_dim, out_dim, bias=False)

    # Creates the PAL Layer
    pal_layer = PAL_Layer(in_dim, out_dim, pal_dim, n_tasks, project_down, project_up)

    # Assigns the weights (making separate copies)
    pal_layer.set_shared_linear(shared_linear.weight.detach().clone(), shared_linear.bias.detach().clone())
    for i in range(n_tasks):
        pal_layer.set_individual_linear(i, individual_linears[i].weight.detach().clone().T, individual_linears[i].bias.detach().clone())
    pal_layer.set_project_down(project_down.weight.detach().clone())
    pal_layer.set_project_up(project_up.weight.detach().clone())

    # Manually computes the output
    x = torch.randn(n_tasks, in_dim)
    y_shared = shared_linear(x)
    x_down = project_down(x)
    y_down = torch.zeros(n_tasks, pal_dim)
    for i in range(n_tasks):
        y_down[i] = individual_linears[i](x_down[i])
    y_individual = project_up(y_down)
    y = y_shared + y_individual
    y = nn.ReLU()(y)

    # Computes the output using the PAL Layer
    y_pal = pal_layer(x)

    # Checks that the outputs are the same
    assert y.shape == y_pal.shape, f"Expected shape {y.shape}, got {y_pal.shape}"
    assert torch.allclose(y, y_pal), f"Expected {y},\ngot {y_pal}"


if __name__ == "__main__":
    test_pal_layer()
