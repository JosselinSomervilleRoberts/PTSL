from mtrl.agent.components.moe_layer import FeedForwardPAL
from mtrl.agent.components.pal_layer import PALLayer
import torch.nn as nn
import torch


def test_feedfoward_pal_shapes():
    """Simply checks that the model does not raise an error and that the output shapes are correct."""
    in_dim, out_dim, hidden_dim, pal_dim, n_layers, n_tasks = 3, 4, 10, 2, 3, 5

    # With shared projection and no residual connections
    model = FeedForwardPAL(
        n_tasks=n_tasks,
        in_features=in_dim,
        out_features=out_dim, 
        num_layers=n_layers,
        hidden_features=hidden_dim,
        pal_features=pal_dim,
        shared_projection=True,
        use_residual_connections=False,
    )
    x = torch.randn(n_tasks, in_dim)
    y = model(x)
    assert y.shape == (n_tasks, out_dim), f"Expected shape {(n_tasks, out_dim)}, got {y.shape}"

    # With shared projection and residual connections
    model = FeedForwardPAL(
        n_tasks=n_tasks,
        in_features=in_dim,
        out_features=out_dim,
        num_layers=n_layers,
        hidden_features=hidden_dim,
        pal_features=pal_dim,
        shared_projection=True,
        use_residual_connections=True,
    )
    x = torch.randn(n_tasks, in_dim)
    y = model(x)
    assert y.shape == (n_tasks, out_dim), f"Expected shape {(n_tasks, out_dim)}, got {y.shape}"

    # Without shared projection and no residual connections
    model = FeedForwardPAL(
        n_tasks=n_tasks,
        in_features=in_dim,
        out_features=out_dim,
        num_layers=n_layers,
        hidden_features=hidden_dim,
        pal_features=pal_dim,
        shared_projection=False,
        use_residual_connections=False,
    )
    x = torch.randn(n_tasks, in_dim)
    y = model(x)
    assert y.shape == (n_tasks, out_dim), f"Expected shape {(n_tasks, out_dim)}, got {y.shape}"


def test_feedforward_pal_number_of_parameters():
    N_TESTS = 1000

    for _ in range(N_TESTS):
        in_dim, out_dim, hidden_dim, pal_dim, n_tasks = torch.randint(1, 100, (5,))
        n_layers = torch.randint(1, 10, (1,))
        shared_projection = torch.randint(0, 2, (1,)).bool()

        model = FeedForwardPAL(
            n_tasks=n_tasks,
            in_features=in_dim,
            out_features=out_dim,
            num_layers=n_layers,
            hidden_features=hidden_dim,
            pal_features=pal_dim,
            shared_projection=shared_projection,
            use_residual_connections=True,
        )

        # Checks that the number of parameters is correct
        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        assert num_params == FeedForwardPAL.compute_number_of_parameters(
            n_tasks=n_tasks,
            in_features=in_dim,
            out_features=out_dim,
            num_layers=n_layers,
            hidden_features=hidden_dim,
            pal_features=pal_dim,
            shared_projection=shared_projection,
        ), f"Expected {num_params}, got {FeedForwardPAL.compute_number_of_parameters(n_tasks=n_tasks, in_features=in_dim, out_features=out_dim, num_layers=n_layers, hidden_features=hidden_dim, pal_features=pal_dim, shared_projection=shared_projection)}"