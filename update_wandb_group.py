import wandb
import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--group", type=str, default="PAL_shared", help="group name")
    parser.add_argument(
        "--project", type=str, default="MTRL10-200k", help="project name"
    )
    parser.add_argument(
        "--new_group", type=str, default="SAC_PTSL", help="new group name"
    )
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    api = wandb.Api()
    for r in api.runs(args.project, filters={"group": args.group}):
        print(f"Updating run: {r.name} (group: {r.group})")
        r.group = args.new_group
        r.name = r.name.replace(args.group, args.new_group)
        print(f"New run name: {r.name} (group: {r.group})")
        r.update()


if __name__ == "__main__":
    main()
