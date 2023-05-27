# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

"""This is the main entry point for the code."""

import hydra
import wandb

from mtrl.app.run import run
from mtrl.utils import config as config_utils
from mtrl.utils.types import ConfigType


def start_wandb(config):
    wandb_name = f"{config.name}_{config.setup.seed}"
    group_wandb = config.name
    config_wandb = {
        "num_tasks": config.agent.multitask.num_envs,
        "agent": config.agent.name,
        "encoder": config.agent.encoder.type_to_select,
        "seed": config.setup.seed,
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
    }
    wandb.init(project="MTRL", name=wandb_name, group=group_wandb, config=config_wandb)


@hydra.main(config_path="config", config_name="config")
def launch(config: ConfigType) -> None:
    config = config_utils.process_config(config)
    start_wandb(config)
    return run(config)


if __name__ == "__main__":
    launch()
