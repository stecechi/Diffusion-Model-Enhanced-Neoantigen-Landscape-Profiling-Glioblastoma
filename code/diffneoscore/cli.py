import argparse
import json
import logging
from pathlib import Path

import torch

from .architecture import DiffNeoScore, DiffNeoScoreConfig
from .checkpointing import set_seed


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(prog="diffneoscore")
    root.add_argument("--log-level", default="INFO")
    commands = root.add_subparsers(dest="command", required=True)
    inspect_model = commands.add_parser("inspect-model")
    inspect_model.add_argument("--output", type=Path)
    train = commands.add_parser("train")
    train.add_argument("--config", type=Path, required=True)
    train.add_argument("--seed", type=int, default=17)
    return root


def main() -> None:
    arguments = parser().parse_args()
    logging.basicConfig(level=getattr(logging, arguments.log_level), format="%(asctime)s %(levelname)s %(message)s")
    if arguments.command == "inspect-model":
        model = DiffNeoScore(DiffNeoScoreConfig())
        payload = {"parameters": sum(parameter.numel() for parameter in model.parameters()), "trainable": sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)}
        encoded = json.dumps(payload, indent=2)
        if arguments.output:
            arguments.output.write_text(encoded + "\n", encoding="utf-8")
        else:
            logging.info(encoded)
        return
    set_seed(arguments.seed)
    config = json.loads(arguments.config.read_text(encoding="utf-8"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DiffNeoScore(DiffNeoScoreConfig(**config.get("model", {}))).to(device)
    logging.info("initialized model with %d parameters on %s", sum(parameter.numel() for parameter in model.parameters()), device)


if __name__ == "__main__":
    main()
