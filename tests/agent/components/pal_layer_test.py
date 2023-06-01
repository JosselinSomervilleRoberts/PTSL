from mtrl.agent.components.pal_layer import PALLayer
from mtrl.agent.components.moe_layer import ModuleList
import numpy as np
import torch.nn as nn
import torch

def test_pal_layer_correctness():
    """Test a PAL Layer to make sure ut works as expected."""
    in_dim, out_dim, pal_dim, n_tasks = 3, 4, 2, 5

    # Creates individually all the components of the PAL Layer
    shared_linear = nn.Linear(in_dim, out_dim)
    individual_linears: ModuleList[nn.Linear] = ModuleList()
    for _ in range(n_tasks):
        individual_linears.append(nn.Linear(pal_dim, pal_dim))
    project_down = nn.Linear(in_dim, pal_dim, bias=False)
    project_up = nn.Linear(pal_dim, out_dim, bias=False)

    # Creates the PAL Layer
    pal_layer = PALLayer(in_dim, out_dim, pal_dim, n_tasks, project_down, project_up)

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
    y_pal_residual, y_pal_residual_down = pal_layer.residual_forward(x)

    # Checks that the outputs are the same
    assert y.shape == y_pal.shape, f"Expected shape {y.shape}, got {y_pal.shape}"
    assert torch.allclose(y, y_pal), f"Expected {y},\ngot {y_pal}"
    assert torch.allclose(y_down, y_pal_residual_down), f"Expected {y_down},\ngot {y_pal_residual_down}"
    assert torch.allclose(y, y_pal_residual), f"Expected {y},\ngot {y_pal_residual}"


def test_pal_layer_skip_project_down():
    in_dim, out_dim, pal_dim, n_tasks = 3, 4, 10, 5

    # Creates the PAL Layer
    pal_layer = PALLayer(in_dim, out_dim, pal_dim, n_tasks)

    # Checks that there is no project down
    assert pal_layer._input_pal_size == in_dim
    
    # Checks that the module still runs
    x = torch.randn(n_tasks, in_dim)
    y = pal_layer(x)
    assert y.shape == (n_tasks, out_dim)

def test_pal_layer_skip_project_up():
    in_dim, out_dim, pal_dim, n_tasks = 3, 1, 10, 5

    # Creates the PAL Layer
    pal_layer = PALLayer(in_dim, out_dim, pal_dim, n_tasks)

    # Checks that there is no project up
    assert pal_layer._output_pal_size == out_dim
    
    # Checks that the module still runs
    x = torch.randn(n_tasks, in_dim)
    y = pal_layer(x)
    assert y.shape == (n_tasks, out_dim)

def test_pal_layer_number_of_parameters():
    N_TESTS = 1000

    for _ in range(N_TESTS):
        in_dim, out_dim, pal_dim, n_tasks = torch.randint(1, 100, (4,))

        # Creates the PAL Layer
        model = PALLayer(in_dim, out_dim, pal_dim, n_tasks)

        # Checks that the number of parameters is correct (with project down and up)
        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        assert num_params == PALLayer.compute_number_of_parameters(in_dim, out_dim, pal_dim, n_tasks, True, True), f"Expected {num_params}, got {PALLayer.compute_number_of_parameters(in_dim, out_dim, pal_dim, n_tasks, True, True)}"
        num_params_before = num_params

        # Checks that the number of parameters is correct (with project down and without project up)
        # To do this we set model.project_up_module.requires_grad to False if it is an nn.Linear
        if isinstance(model.project_up_module, nn.Linear):
            for param in model.project_up_module.parameters():
                param.requires_grad = False
        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(num_params, num_params_before)
        assert num_params == PALLayer.compute_number_of_parameters(in_dim, out_dim, pal_dim, n_tasks, True, False), f"Expected {num_params}, got {PALLayer.compute_number_of_parameters(in_dim, out_dim, pal_dim, n_tasks, True, False)}"

        # Checks that the number of parameters is correct (without project down and without project up)
        # To do this we set model.project_down_module.requires_grad to False if it is an nn.Linear
        if isinstance(model.project_down_module, nn.Linear):
            for param in model.project_down_module.parameters():
                param.requires_grad = False
        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        assert num_params == PALLayer.compute_number_of_parameters(in_dim, out_dim, pal_dim, n_tasks, False, False), f"Expected {num_params}, got {PALLayer.compute_number_of_parameters(in_dim, out_dim, pal_dim, n_tasks, False, True)}"

        # Checks that the number of parameters is correct (without project down and with project up)
        if isinstance(model.project_up_module, nn.Linear):
            for param in model.project_up_module.parameters():
                param.requires_grad = True
        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        assert num_params == PALLayer.compute_number_of_parameters(in_dim, out_dim, pal_dim, n_tasks, False, True), f"Expected {num_params}, got {PALLayer.compute_number_of_parameters(in_dim, out_dim, pal_dim, n_tasks, False, False)}"