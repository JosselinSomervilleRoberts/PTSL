[![CircleCI](https://circleci.com/gh/facebookresearch/mtrl.svg?style=svg&circle-token=8cc8eb1b9666a65e27a21c39b5d5398744365894)](https://circleci.com/gh/facebookresearch/mtrl)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://github.com/facebookresearch/mtrl/blob/main/LICENSE)
[![Python 3.6+](https://img.shields.io/badge/python-3.6+-blue.svg)](https://www.python.org/downloads/release/python-360/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Zulip Chat](https://img.shields.io/badge/zulip-join_chat-brightgreen.svg)](https://mtenv.zulipchat.com)

# This is a fork of MTRL

## Installation steps
We recommend using `conda` to install this as we were not able to install the package using facebook's isntructions. Instead we provide a custom `environment.yml` to install it with `conda`.
```bash
conda env create -f environment.yml
conda activate mtrl
pip install git+https://github.com/Farama-Foundation/Metaworld.git@af8417bfc82a3e249b4b02156518d775f29eb28
pip install "mtenv[metaworld]"
pip install git+https://github.com/JosselinSomervilleRoberts/JossPythonToolbox.git
pip install wandb
```

You will need to install Mujoco. If you are on an AWS EC2 instance, you can run our script: `aws_setup.sh` that should handle everything for you.

You can check your installation by running:
```bash
PYTHONPATH=. python3 -u main.py setup=metaworld agent=state_sac env=metaworld-mt10 agent.multitask.num_envs=10 agent.multitask.should_use_disentangled_alpha=True
```
You should see something like this (after a few minutes):
```
| train | E: 5270 | S: 790500 | D: 2.7 s | Su: 0.6000 | BR: 1.1579 | ALOSS: -129.9481 | CLOSS: 48.9331 | R_0: 487.5569 | R_1: 8.9984 | R_2: 158.9726 | R_3: 186.7860 | R_4: 17.6790 | R_5: 179.2791 | R_6: 224.1127 | R_7: 119.9955 | R_8: 100.4628 | R_9: 148.0436 | Su_0: 1.0000 | Su_1: 0.0000 | Su_2: 1.0000 | Su_3: 1.0000 | Su_4: 0.0000 | Su_5: 1.0000 | Su_6: 1.0000 | Su_7: 0.0000 | Su_8: 0.0000 | Su_9: 1.0000 | ENV_0: 0 | ENV_1: 1 | ENV_2: 2 | ENV_3: 3 | ENV_4: 4 | ENV_5: 5 | ENV_6: 6 | ENV_7: 7 | ENV_8: 8 | ENV_9: 9
| train | E: 5271 | S: 790650 | D: 2.7 s | Su: 0.7000 | BR: 1.1659 | ALOSS: -131.8787 | CLOSS: 45.3631 | R_0: 458.9632 | R_1: 294.8278 | R_2: 88.5741 | R_3: 80.9546 | R_4: 328.5237 | R_5: 0.4047 | R_6: 162.7022 | R_7: 227.9077 | R_8: 79.3807 | R_9: 151.6023 | Su_0: 1.0000 | Su_1: 1.0000 | Su_2: 1.0000 | Su_3: 1.0000 | Su_4: 1.0000 | Su_5: 0.0000 | Su_6: 1.0000 | Su_7: 0.0000 | Su_8: 0.0000 | Su_9: 1.0000 | ENV_0: 0 | ENV_1: 1 | ENV_2: 2 | ENV_3: 3 | ENV_4: 4 | ENV_5: 5 | ENV_6: 6 | ENV_7: 7 | ENV_8: 8 | ENV_9: 9
```

# MTRL
Multi Task RL Algorithms

## Contents

1. [Introduction](#Introduction)

2. [Setup](#Setup)

3. [Usage](#Usage)

4. [Documentation](#Documentation)

5. [Contributing to MTRL](#Contributing-to-MTRL)

6. [Community](#Community)

7. [Acknowledgements](#Acknowledgements)

## Introduction

MTRL is a library of multi-task reinforcement learning algorithms. It has two main components:

* [Building blocks](https://github.com/facebookresearch/mtrl/tree/main/mtrl/agent/components) and [agents](https://github.com/facebookresearch/mtrl/tree/main/mtrl/agent) that implement the multi-task RL algorithms.

* [Experiment setups](https://github.com/facebookresearch/mtrl/tree/main/mtrl/experiment) that enable training/evaluation on different setups. 

Together, these two components enable use of MTRL across different environments and setups.

### List of publications & submissions using MTRL (please create a pull request to add the missing entries):

* [Learning Robust State Abstractions for Hidden-Parameter Block MDPs](https://arxiv.org/abs/2007.07206)
* [Multi-Task Reinforcement Learning with Context-based Representations](https://arxiv.org/abs/2102.06177)
    *  We use the `af8417bfc82a3e249b4b02156518d775f29eb289` commit for the MetaWorld environments for our experiments.

### License

* MTRL uses [MIT License](https://github.com/facebookresearch/mtrl/blob/main/LICENSE).

* [Terms of Use](https://opensource.facebook.com/legal/terms)

* [Privacy Policy](https://opensource.facebook.com/legal/privacy)

### Citing MTRL

If you use MTRL in your research, please use the following BibTeX entry:
```
@Misc{Sodhani2021MTRL,
  author =       {Shagun Sodhani and Amy Zhang},
  title =        {MTRL - Multi Task RL Algorithms},
  howpublished = {Github},
  year =         {2021},
  url =          {https://github.com/facebookresearch/mtrl}
}
```

## Setup

* Clone the repository: `git clone git@github.com:facebookresearch/mtrl.git`.

* Install dependencies: `pip install -r requirements/dev.txt`

## Usage

* MTRL supports 8 different multi-task RL algorithms as described [here](https://mtrl.readthedocs.io/en/latest/pages/tutorials/overview.html).

* MTRL supports multi-task environments using [MTEnv](https://github.com/facebookresearch/mtenv). These environments include [MetaWorld](https://meta-world.github.io/) and multi-task variants of [DMControl Suite](https://github.com/deepmind/dm_control)

* Refer the [tutorial](https://mtrl.readthedocs.io/en/latest/pages/tutorials/overview.html) to get started with MTRL.

## Documentation

[https://mtrl.readthedocs.io](https://mtrl.readthedocs.io)

## Contributing to MTRL

There are several ways to contribute to MTRL.

1. Use MTRL in your research.

2. Contribute a new algorithm. We currently support [8 multi-task RL algorithms](https://mtrl.readthedocs.io/en/latest/pages/algorithms/supported.html) and are looking forward to adding more environments.

3. Check out the [good-first-issues](https://github.com/facebookresearch/mtrl/pulls?q=is%3Apr+is%3Aopen+label%3A%22good+first+issue%22) on GitHub and contribute to fixing those issues.

4. Check out additional details [here](https://github.com/facebookresearch/mtrl/blob/main/.github/CONTRIBUTING.md).

## Community

Ask questions in the chat or github issues:
* [Chat](https://mtenv.zulipchat.com)
* [Issues](https://github.com/facebookresearch/mtrl/issues)

## Acknowledgements

* Our implementation of SAC is inspired by Denis Yarats' implementation of [SAC](https://github.com/denisyarats/pytorch_sac).
* Project file pre-commit, mypy config, towncrier config, circleci etc are based on same files from [Hydra](https://github.com/facebookresearch/hydra).
