# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

"""This is the main entry point for the code."""

import hydra
import wandb
import time
import os

from toolbox.aws import shutdown

from mtrl.app.run import run
from mtrl.utils import config as config_utils
from mtrl.utils.types import ConfigType
from toolbox.printing import print_visible


def start_wandb(config, seed: int = -1):
    wandb_seed = seed if seed >= 0 else config.setup.seed
    wandb_name = f"{config.name}_{wandb_seed}"
    group_wandb = config.name
    config_wandb = {
        "num_tasks": config.agent.multitask.num_envs,
        "agent": config.agent.name,
        "encoder": config.agent.encoder.type_to_select,
        "seed": wandb_seed,
        "agent/encoder_feature_dim": config.agent.encoder_feature_dim,
        "agent/num_layers": config.agent.num_layers,
        "agent/num_filters": config.agent.num_filters,
        "actor/num_layer": config.agent.actor.num_layers,
        "actor/hidden_dim": config.agent.actor.hidden_dim,
        "num_train_steps": config.experiment.num_train_steps,
        "eval_freq": config.experiment.eval_freq,
        "num_eval_episodes": config.experiment.num_eval_episodes,
        "lr/actor": config.agent.optimizers.actor.lr,
        "lr/critic": config.agent.optimizers.critic.lr,
        "lr/alpha": config.agent.optimizers.alpha.lr,
        "lr/decoder": config.agent.optimizers.decoder.lr,
        "lr/encoder": config.agent.optimizers.encoder.lr,
        "batch_size": config.replay_buffer.batch_size,
        "pal/dim": config.agent.multitask.pal_cfg.pal_dim,
        "pal/shared": config.agent.multitask.pal_cfg.shared_projection,
        "pal/residual": config.agent.multitask.pal_cfg.use_residual_connections,
    }
    wandb.init(project=f"MTRL{config.agent.multitask.num_envs}", name=wandb_name, group=group_wandb, config=config_wandb)

def launch_one_seed(config, seed: int, time_start: int = -1):
    if time_start < 0: time_start = time.time()

    try:
        # RUn "mv logs/* logs_saved/"
        print_visible("Starting seed " + str(seed))
        os.system("rm -r -f /home/ubuntu/mtrl/logs")
        start_wandb(config, seed=seed)
        run(config, seed=seed)
    except Exception as e:
        # If it has been running for less than 5 minutes, then it is probably a bug
        # Otherwise, it is probably a timeout, so shutdown the instance
        if time.time() - time_start < 5 * 60:
            raise e
        else:
            print(f"Exception: {e}")
            print("Timeout, shutting down")
            wandb.finish()
            shutdown()
            return
        
    wandb.finish()
    return

@hydra.main(config_path="config", config_name="config")
def launch(config: ConfigType) -> None:
    seed_ref = config.setup.seed
    config = config_utils.process_config(config)
    time_start = time.time()

    for seed_inc in range(config.num_seeds):
        seed = seed_ref + seed_inc
        launch_one_seed(config, seed=seed, time_start=time_start)
    shutdown()
    return


if __name__ == "__main__":
    launch()
