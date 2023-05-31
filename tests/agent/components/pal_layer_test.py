from mtrl.agent.components.pal_layer import PALLayer
import torch.nn as nn
import torch

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

    # Checks that the outputs are the same
    assert y.shape == y_pal.shape, f"Expected shape {y.shape}, got {y_pal.shape}"
    assert torch.allclose(y, y_pal), f"Expected {y},\ngot {y_pal}"
