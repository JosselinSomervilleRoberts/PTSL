import torch
import torch.nn as nn
from typing import Optional, Tuple
from toolbox.printing import debug as print_debug
from toolbox.printing import str_with_color
from mtrl.agent.ds.mt_obs import MTObs


class PALLayer(nn.Module):
    """
    Implementation of a PAL Layer.

    A given output x will go through a shqred linear layer SL
    and a low-rank task specific linear layer TL(i).
    There are to matrices D and U to down project x to the input dimension
    of TL(i) and to up project the output of TL(i) to match the final
    output size. An activation function is added at the end.
    So, y = activation( SL(x) + U @ TL(i)( D @ x ) )
    
    A twist is that depending on the input size, it might take less parameters
    to directly have TL go from input_size to pal_size. If this is the case, then
    the down projection D will just be the identity.
    The similar concept applies for the upsampling U.

    This implementation assumes that the batch size is equal to the number of tasks.
    """

    @staticmethod
    def get_project_down_module(input_size: int, pal_size: int) -> nn.Linear:
        return nn.Linear(in_features=input_size, out_features=pal_size, bias=False)
    
    @staticmethod
    def get_project_up_module(output_size: int, pal_size: int) -> nn.Linear:
        return nn.Linear(in_features=pal_size, out_features=output_size, bias=False)

    def __init__(self,
                 input_size: int,
                 output_size: int,
                 pal_size: int,
                 n_tasks: int,
                 project_down_module: Optional[nn.Linear] = None,
                 project_up_module: Optional[nn.Linear] = None,
                 activation: nn.Module = nn.ReLU(),
                 residual_mode: str = "none",
                 residual_project_module: Optional[nn.Linear] = None,
                 residual_alpha: Optional[nn.Parameter] = None,
                 debug: bool = False):
        super(PALLayer, self).__init__()
        self.activation = activation
        self.n_tasks = n_tasks
        self.pal_size = pal_size
        self.input_size = input_size
        self.output_size = output_size
        self.indices = None
        self._residual_mode = residual_mode
        assert self._residual_mode in ["none", "sum", "linear", "project"], f"Invalid residual mode {self._residual_mode}. Modes accepted are: none, sum, linear, project"
        self._debug = debug

        # Shared Linear, it is a vanilla Linear layer
        self.shared_linear = nn.Linear(input_size, output_size)

        # Project Down
        if project_down_module is None:
            # First figure out if it is better to project down or simply skip the projection
            n_parameters_with_projection: int = input_size * pal_size + n_tasks * (pal_size + 1) * pal_size
            n_parameteres_without_projection: int = n_tasks * (input_size + 1) * pal_size
            self._project_down = (n_parameters_with_projection <= n_parameteres_without_projection)
            self. _input_pal_size = pal_size if self._project_down else input_size
            if self._project_down:
                self.project_down_module = PALLayer.get_project_down_module(input_size=input_size, pal_size=pal_size)
            else:
                self.project_down_module = nn.Identity()
            if self._debug:
                print("Project down decision:")
                print(f" - Num. params. with projection: {n_parameters_with_projection}")
                print(f" - Num. params. without projection: {n_parameteres_without_projection}")
                print(f"-> Decision: Use projection? {self._project_down}\n")
        else:
            self.project_down_module = project_down_module
            self._project_down = True
            self. _input_pal_size = pal_size
            assert self.project_down_module.in_features == input_size, f"Expected project_down_module to have input features of size {input_size}, got {self.project_down_module.in_features}"
            assert self.project_down_module.out_features == pal_size, f"Expected project_down_module to have output features of size {pal_size}, got {self.project_down_module.out_features}"

        # Project Up
        if project_up_module is None:
            # First figure out if it is better to project up or simply skip the projection
            n_parameters_with_projection: int = output_size * pal_size + n_tasks * (self._input_pal_size + 1) * pal_size
            n_parameters_without_projection: int = n_tasks * (self._input_pal_size + 1) * output_size
            self._project_up = (n_parameters_with_projection <= n_parameters_without_projection)
            self._output_pal_size = pal_size if self._project_up else output_size
            if self._project_up:
                self.project_up_module = PALLayer.get_project_up_module(output_size=output_size, pal_size=pal_size)
            else:
                self.project_up_module = nn.Identity()
            if self._debug:
                print("Project up decision:")
                print(f" - Num. params. with projection: {n_parameters_with_projection}")
                print(f" - Num. params. without projection: {n_parameters_without_projection}")
                print(f"-> Decision: Use projection? {self._project_up}\n")
        else:
            self.project_up_module = project_up_module
            self._project_up = True
            self._output_pal_size = pal_size
            assert self.project_up_module.in_features == pal_size, f"Expected project_up_module to have input features of size {pal_size}, got {self.project_up_module.in_features}"
            assert self.project_up_module.out_features == output_size, f"Expected project_up_module to have output features of size {output_size}, got {self.project_up_module.out_features}"

        # Individual Linear. This represents n_tasks linear layers of _input_al_size to _output_pal_size
        self.individual_linears_weight = nn.Parameter(torch.randn(n_tasks, self._input_pal_size, self._output_pal_size))
        self.individual_linears_bias = nn.Parameter(torch.zeros(n_tasks, self._output_pal_size))

        # Residual
        self.residual_project_module = None
        if self._residual_mode == "linear":
            if residual_alpha is None:
                self.residual_alpha = nn.Parameter(torch.ones(3) / 3.0)
            else:
                self.residual_alpha = residual_alpha
        elif self._residual_mode == "project" and self._project_down and self._input_pal_size == pal_size:
            if residual_project_module is None:
                self.residual_project_module = PALLayer.get_project_down_module(input_size=3*pal_size, pal_size=pal_size)
            else:
                self.residual_project_module = residual_project_module

    def set_shared_linear(self, weights: torch.Tensor, bias: torch.Tensor) -> None:
        assert weights.shape == (self.output_size, self.input_size), f"Expected shape for shared weight {(self.output_size, self.input_size)}, got {weights.shape}"
        assert bias.shape == (self.output_size,), f"Expected shape for shared bias {(self.output_size,)}, got {bias.shape}"
        self.shared_linear.weight = nn.Parameter(weights)
        self.shared_linear.bias = nn.Parameter(bias)

    def set_individual_linear(self, task_id: int, weights: torch.Tensor, bias: torch.Tensor) -> None:
        assert weights.shape == (self._input_pal_size, self._output_pal_size), f"Expected shape for individual weight {task_id} {(self.pal_size, self.pal_size)}, got {weights.shape}"
        assert bias.shape == (self.pal_size,), f"Expected shape for individual bias {task_id} {(self.pal_size,)}, got {bias.shape}"
        self.individual_linears_weight.data[task_id] = weights
        self.individual_linears_bias.data[task_id] = bias

    def set_project_down(self, weights: torch.Tensor) -> None:
        assert weights.shape == (self._input_pal_size, self.input_size), f"Expected shape for project down {(self.pal_size, self.input_size)}, got {weights.shape}"
        self.project_down_module.weight = nn.Parameter(weights)

    def set_project_up(self, weights: torch.Tensor) -> None:
        assert weights.shape == (self.output_size, self._input_pal_size), f"Expected shape for project up {(self.output_size, self.pal_size)}, got {weights.shape}"
        self.project_up_module.weight = nn.Parameter(weights)

    def set_residual_project(self, weights: torch.Tensor) -> None:
        assert self._residual_mode == "project", f"Residual mode is not project, it is {self._residual_mode}"
        assert self.residual_project_module is not None, "Residual project module is None"
        assert weights.shape == (self.pal_size, 3*self.pal_size), f"Expected shape for residual project {(self.pal_size, 3*self.pal_size)}, got {weights.shape}"
        self.residual_project_module.weight = nn.Parameter(weights)

    def set_residual_alpha(self, alpha: torch.Tensor) -> None:
        assert self._residual_mode == "linear", f"Residual mode is not linear, it is {self._residual_mode}"
        assert alpha.shape == (3,), f"Expected shape for residual alpha {(3,)}, got {alpha.shape}"
        self.alpha = nn.Parameter(alpha)

    def set_debug(self, debug: bool) -> None:
        self._debug = debug

    def set_indices(self, indices: torch.Tensor) -> None:
        self.indices = indices.reshape(-1)

    def summary(self, prefix: str = "") -> str:
        """Summary of the PALLayer.

        Args:
            prefix (str, optional): prefix to add to the summary before each line.
                Defaults to "".

        Returns:
            str: summary of the PALLayer.
        """
        summary: str = ""
        num_parameters = sum(p.numel() for p in self.parameters() if p.requires_grad)
        summary += f"{prefix}PALLayer " + str_with_color(f"({num_parameters} parameters)", "purple") + "\n"
        summary += f"{prefix}    Shared Linear: {self.shared_linear}\n"
        if self._project_down:
            summary += f"{prefix}    Project Down: {self.project_down_module}\n"
        else:
            summary += f"{prefix}    Project Down: None\n"
        if self._project_up:
            summary += f"{prefix}    Project Up: {self.project_up_module}\n"
        else:
            summary += f"{prefix}    Project Up: None\n"
        summary += f"{prefix}    Individual Linears:\n"
        summary += f"{prefix}        Weight: Tensor ({self.individual_linears_weight.shape})\n"
        summary += f"{prefix}        Bias: Tensor ({self.individual_linears_bias.shape})\n"
        return summary
    
    def __repr__(self) -> str:
        return self.summary()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # If x is a MTObs, then we need to extract the env_obs
        if isinstance(x, MTObs):
            x = x.env_obs
            
        assert self._residual_mode == "none", f"Residual mode is not none, it is {self._residual_mode}. You should use residual_forward instead."
        if self.indices is None:
            assert x.shape == (self.n_tasks, self.input_size), f"Since no indices are specified, expected shape {(self.n_tasks, self.input_size)}, got {x.shape}"
        else:
            assert x.shape == (self.indices.shape[0], self.input_size), f"Since indices are specified, expected shape {(self.indices.shape[0], self.input_size)}, got {x.shape}"
        if self._debug: print_debug(x)

        # Shared linear
        y_shared = self.shared_linear(x)
        if self._debug: print_debug(y_shared)

        # Individual linears
        # 1. Project down
        x_down = self.project_down_module(x)
        if self._debug: print_debug(x_down)

        # 2. Compute X @ Wi for all i efficiently using broadcasting
        x_down = x_down.unsqueeze(1)
        weights = self.individual_linears_weight[self.indices] if self.indices is not None else self.individual_linears_weight
        bias = self.individual_linears_bias[self.indices] if self.indices is not None else self.individual_linears_bias
        y_down = torch.matmul(x_down, weights)
        y_down = y_down.squeeze(1) + bias
        if self._debug: print_debug(y_down)

        # 3. Project up
        y_individual = self.project_up_module(y_down)
        if self._debug: print_debug(y_individual)

        # Sum and apply activation
        y = y_shared + y_individual
        y = self.activation(y)
        if self._debug: print_debug(y)
        return y
    
    def residual_forward(self, x: torch.Tensor, residual_x_down: Optional[torch.Tensor] = None, residual_y_down: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Similar to forward but uses the residual_x_down and residual_y_down."""
        # If x is a MTObs, then we need to extract the env_obs
        if isinstance(x, MTObs):
            x = x.env_obs
            
        assert residual_x_down is None or residual_x_down.shape == (x.shape[0], self._input_pal_size), f"Expected shape for residual_x_down {(x.shape[0], self._input_pal_size)}, got {residual_x_down.shape}"
        assert residual_y_down is None or residual_y_down.shape == (x.shape[0], self._input_pal_size), f"Expected shape for residual_y_down {(x.shape[0], self._input_pal_size)}, got {residual_y_down.shape}"
        
        if self.indices is None:
            assert x.shape == (self.n_tasks, self.input_size), f"Since no indices are specified, expected shape {(self.n_tasks, self.input_size)}, got {x.shape}"
        else:
            assert x.shape == (self.indices.shape[0], self.input_size), f"Since indices are specified, expected shape {(self.indices.shape[0], self.input_size)}, got {x.shape}"
        if self._debug: print_debug(x)

        # Shared linear
        y_shared = self.shared_linear(x)
        if self._debug: print_debug(y_shared)

        # Individual linears
        # 1. Project down
        x_down = self.project_down_module(x)
        if self._debug: print_debug(x_down)

        # If we use the sum residual node then we simply add x_down, residual_x_down and residual_y_down
        x_down_with_residual = x_down
        if residual_x_down is not None and residual_y_down is not None:
            if self._residual_mode == "sum":
                x_down_with_residual = x_down + residual_x_down + residual_y_down
            elif self._residual_mode == "linear":
                x_down_with_residual = self.residual_alpha[0] * x_down + self.residual_alpha[1] * residual_x_down + self.residual_alpha[2] * residual_y_down
            elif self._residual_mode == "project":
                input_projection = torch.cat((x_down, residual_x_down, residual_y_down), dim=1)
                if self._debug: print_debug(input_projection)
                x_down_with_residual = self.residual_project_module(input_projection)
            if self._debug: print_debug(x_down_with_residual)

        # 2. Compute X @ Wi for all i efficiently using broadcasting
        x_down_with_residual = x_down_with_residual.unsqueeze(1)
        weights = self.individual_linears_weight[self.indices] if self.indices is not None else self.individual_linears_weight
        bias = self.individual_linears_bias[self.indices] if self.indices is not None else self.individual_linears_bias
        y_down = torch.matmul(x_down_with_residual, weights)
        y_down = y_down.squeeze(1) + bias
        if self._debug: print_debug(y_down)

        # 3. Project up
        y_individual = self.project_up_module(y_down)
        if self._debug: print_debug(y_individual)

        # Sum and apply activation
        y = y_shared + y_individual
        y = self.activation(y)
        if self._debug: print_debug(y)
        return y, y_down, x_down
    
    @staticmethod
    def compute_number_of_parameters(input_size: int, output_size: int, pal_size: int, n_tasks: int, project_down: bool, project_up: bool, must_project_down: bool = False, must_project_up: bool = False) -> int:
        """
        Computes the number of parameters of a PAL Layer with the given parameters.
        Args:
            input_size: Size of the input.
            output_size: Size of the output.
            pal_size: Size of the PAL layer.
            n_tasks: Number of tasks.
            project_down: Whether to create a projection matrix or not.
            project_up: Whether to create a projection matrix or not.
            must_project_down: Whether to force the projection down or not.
            must_project_up: Whether to force the projection up or not.
        Returns:
            The number of parameters of the PAL Layer.
        """

        n_parameters = 0

        # Shared Linear
        n_parameters += output_size * input_size + output_size

        # Project Down
        # Figure out if it is better to project down or simply skip the projection
        n_parameters_with_projection = input_size * pal_size + n_tasks * (pal_size + 1) * pal_size
        n_parameteres_without_projection = n_tasks * (input_size + 1) * pal_size
        if n_parameters_with_projection <= n_parameteres_without_projection or must_project_down:
            if project_down: n_parameters += input_size * pal_size
            input_pal_size = pal_size
        else:
            input_pal_size = input_size


        # Project Up
        # Figure out if it is better to project up or simply skip the projection
        n_parameters_with_projection = output_size * pal_size + n_tasks * (input_pal_size + 1) * pal_size
        n_parameters_without_projection = n_tasks * (input_pal_size + 1) * output_size
        if n_parameters_with_projection <= n_parameters_without_projection or must_project_up:
            if project_up: n_parameters += output_size * pal_size
            output_pal_size = pal_size
        else:
            output_pal_size = output_size

        # Individual Linear
        n_parameters += n_tasks * input_pal_size * output_pal_size + n_tasks * output_pal_size

        return n_parameters
