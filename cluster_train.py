#!/usr/bin/env python3
import hydra
from train import TrainConfig


@hydra.main(version_base="1.3", config_path="./cfgs", config_name="train")
def cluster_train(cfg: TrainConfig):
    """Script for submitting SLURM jobs using hydra's submitit plugin.
    For example, using commands like:

        python train.py -m ++seed=1,2,3,4,5 ++agent.use_rew_loss=True,False
    """
    import sys
    import traceback

    import train

    # This main is used to circumvent a bug in Hydra
    # See https://github.com/facebookresearch/hydra/issues/2664

    try:
        train.train(cfg)
    except BaseException:
        traceback.print_exc(file=sys.stderr)
        raise
    finally:
        # fflush everything
        sys.stdout.flush()
        sys.stderr.flush()


if __name__ == "__main__":
    cluster_train()  # pyright: ignore
