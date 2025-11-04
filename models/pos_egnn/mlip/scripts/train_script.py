import argparse

import torch
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.strategies import DDPStrategy

from geodite.model import GeoditeModule
from geodite.utils.callbacks import CALLBACKS_MAPPING
from geodite.utils.config_parser import ConfigParser
from geodite.utils.training_utils import (
    get_checkpoint_folder,
    get_dataloader,
    get_exp_name,
    set_logger,
    set_profiler,
)

torch.set_float32_matmul_precision("highest")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train model with optional checkpoint reload")
    parser.add_argument("--config", type=str, required=True, help="Path to 'config.yaml' file.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to 'checkpoint' folder.")
    parser.add_argument("--dataset", type=str, required=True, help="Path to dataset folder.")
    parser.add_argument(
        "--aim",
        type=str,
        required=False,
        default=".aim",
        help="(optional) Path to aim server (i.e. /path OR aim://IP_ADDRESS:PORT_NUMBER OR route URL ('aim://x.y.z').",
    )
    parser.add_argument("--constants", type=str, default=None, required=False, help="(optional) Path to 'constants.yaml' file.")
    parser.add_argument(
        "--experiment",
        type=str,
        required=False,
        help="(optional) The experiment Name. In case one is not provided, a random one will be assigned.",
    )
    parser.add_argument("--base_model", type=str, required=False, help=" (optional) Initialized weights")
    args = parser.parse_args()

    exp_name = get_exp_name(args)
    config = ConfigParser(args)
    logger = set_logger(config=config, exp_name=exp_name)
    profiler = set_profiler(config=config, exp_name=exp_name)
    ckpt_path, ckpt, ckpt_exists = get_checkpoint_folder(config, exp_name)
    seed_everything(config["general"]["seed"])

    if ckpt_exists:
        assert args.base_model is None
        print(f"Resuming from checkpoint: {ckpt}\nIgnoring current config")
        model = GeoditeModule.load_from_checkpoint(ckpt, strict=False)
    elif args.base_model:
        print(f"Starting from pre-initialized weights: {args.base_model}")
        model = GeoditeModule(config.config)
        sd = torch.load(args.base_model)["state_dict"]

        model.load_state_dict(sd, strict=False)

        for k, v in model.named_parameters():
            if k in sd.keys():
                assert torch.equal(v.cpu(), sd[k].cpu())
            else:
                print(f"unable to load weight {k}")
    else:
        print("No checkpoint found. Starting from scratch.")
        model = GeoditeModule(config.config)

    callbacks = []
    for callback_config in config["callbacks"]:
        if callback_config["name"] == "checkpoint":
            callbacks.append(CALLBACKS_MAPPING[callback_config["name"]](dirpath=ckpt_path, **callback_config["args"]))
        elif callback_config["name"] == "early_stopping":
            callbacks.append(
                CALLBACKS_MAPPING[callback_config["name"]](monitor="Total loss/Validation", mode="min", **callback_config["args"])
            )
        else:
            callbacks.append(CALLBACKS_MAPPING[callback_config["name"]](**callback_config["args"]))

    dataloader = get_dataloader(config.config)

    if config.get("trainer", {}).get("strategy") == "ddp":
        import datetime

        trainer_config = config["trainer"].copy()
        trainer_config["strategy"] = DDPStrategy(timeout=datetime.timedelta(hours=10))
    else:
        trainer_config = config["trainer"]

    if "batch_size" in config["dataset"].keys():
        print("Using fixed batch size.")
        auto_inject_sampler = True
    elif "max_edges" in config["dataset"].keys():
        print("Using dynamic sampler.")
        auto_inject_sampler = False

    trainer = Trainer(
        logger=logger,
        callbacks=callbacks,
        deterministic=True,
        use_distributed_sampler=auto_inject_sampler,  # To work with custom distributed batch sampler
        profiler=profiler,
        benchmark=True,
        **trainer_config,
    )

    trainer.fit(model, dataloader, ckpt_path=ckpt if ckpt_exists else None)
