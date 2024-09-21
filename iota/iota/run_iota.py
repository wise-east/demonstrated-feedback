# usage: iota <config file> {additional at-runtime specified arguments in IotaArguments}
# main script for running iota
import os
from pathlib import Path
import yaml
from dataclasses import asdict
from loguru import logger
import torch
from datetime import datetime
import pytz
import json
import sys
from .utils import set_absolute_path, H4ArgumentParser, get_git_hash, PACKAGE_DIR
from .iota_trainer import IotaTrainer, IotaArguments
from .eval_utils import get_run_alias


def run_iota(args):

    ### Setup arguments
    model_name_or_path = args.model_name_or_path

    ### setup output dirs and logging

    # get pacific time
    pacific = pytz.timezone("US/Pacific")
    date_str = datetime.now(pacific).strftime("%Y-%m-%d_%H-%M-%S")

    args.output_dir = set_absolute_path(args.output_dir, args.package_dir)
    args.date = date_str
    # group by model name
    args.output_dir = str(
        args.output_dir
        / model_name_or_path
        / args.dataset
        / args.mode
        / str(args.author_key)
        / date_str
    )
    os.makedirs(args.output_dir, exist_ok=True)

    #### save config file to output dir
    with open(Path(args.output_dir) / "config.yaml", "w") as f:
        yaml.dump(asdict(args), f)

    global logger

    logging_dir = set_absolute_path(args.logging_dir, args.package_dir)
    os.makedirs(logging_dir, exist_ok=True)
    logger.add(sys.stdout, format="{time} - {level} - {message}", filter="sub.module")
    logger.add(logging_dir / f"iota_{date_str}.log", rotation="10 MB")
    logger.add(
        Path(args.output_dir) / f"iota_{date_str}.log", rotation="10 MB"
    )  # also add to save dir for convenience

    logger.info(f"Running IOTA with config: {args}")

    ### setup iota trainer
    if args.load_from_checkpoint:
        logger.info(f"Loading from checkpoint: {args.load_from_checkpoint}")
        trainer = IotaTrainer.from_checkpoint(args.load_from_checkpoint)
    else:
        trainer = IotaTrainer(args)

    # test eval
    if trainer.args.mode == "iota":
        logger.info("Testing eval")
        trainer.eval(trainer.val_data, split="val")

    ### train iota
    if trainer.args.do_train:
        trainer.train()

    if trainer.args.do_predict:
        trainer.eval_on_test()


def main():
    parser = H4ArgumentParser((IotaArguments))
    args = parser.parse()
    try:
        run_iota(args)
    except Exception as e:
        logger.exception(
            "An unhandled exception occurred."
        )  # Automatically logs the stack trace


if __name__ == "__main__":
    main()
