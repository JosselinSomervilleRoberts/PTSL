import argparse


def parse_args():
    parser = argparse.ArgumentParser(
        description="Count the number of parameters in a PyTorch model."
    )
    parser.add_argument(
        "--num_layers", type=int, default=3, help="Number of layers in the model."
    )
    parser.add_argument(
        "--hidden_dim", type=int, default=400, help="Hidden dimension of the model."
    )
    parser.add_argument(
        "--pal_dim", type=int, default=50, help="Hidden dimension of the model."
    )
    parser.add_argument(
        "--num_envs", type=int, default=10, help="Number of environments."
    )
    parser.add_argument(
        "--input_dim", type=int, default=100, help="Input dimension."
    )
    parser.add_argument(
        "--no_initial_project_down", action="store_true", help="No initial projection down."
    )
    parser.add_argument(
        "--no_encoder", action="store_true", help="No encoder."
    )
    args = parser.parse_args()
    return args


def main(args):
    num_layers = args.num_layers
    hidden_dim = args.hidden_dim
    pal_dim = args.pal_dim
    num_envs = args.num_envs

    num_params = 0
    if not args.no_encoder:
        num_params += 18104 * 3  # There are 3 Encoders

    # PAL
    for output_dim, input_dim in zip([8, 1, 1, 1, 1], [0, 4, 4, 4, 4]):
        input_dim += args.input_dim
        pal_params = 0
        pal_params = (num_layers - args.no_initial_project_down) * (
            pal_dim**2 * num_envs + pal_dim * num_envs
        )  # Set of individual layers
        pal_params += (
            pal_dim * output_dim * num_envs + output_dim * num_envs
        )  # Last individual layer
        pal_params += (num_layers - 1) * (
            hidden_dim**2 + hidden_dim
        )  # Shared layers (middles)
        pal_params += input_dim * hidden_dim + hidden_dim  # First shared layer
        pal_params += hidden_dim * output_dim + output_dim  # Last shared layer
        # Projections
        if not args.no_initial_project_down:
            print("Initial projection down")
            pal_params += input_dim * pal_dim  # First projection
        else:
            print("No initial projection down")
            pal_params += input_dim * pal_dim * num_envs + pal_dim * num_envs # First individual layer
        pal_params += 2 * pal_dim * hidden_dim  # Middle projections
        print("PAL parameters: ", pal_params)

        # Total num params
        num_params += pal_params
    num_params += num_envs  # Log alpha
    print(f"Number of parameters: {num_params}")


if __name__ == "__main__":
    args = parse_args()
    main(args)
