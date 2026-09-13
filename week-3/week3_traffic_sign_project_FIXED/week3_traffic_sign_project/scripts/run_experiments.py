#!/usr/bin/env python3
import argparse
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import load_config
from src.experiments import run_all
from src.utils import seed_everything, choose_device


def main():
    p = argparse.ArgumentParser(description="Run controlled GTSRB traffic-sign experiments")
    p.add_argument("--config", default="configs/experiments.yaml")
    args = p.parse_args()
    cfg = load_config(args.config)
    seed_everything(cfg["seed"])
    cfg["device"] = choose_device(cfg["device"])
    print(f"Using device: {cfg['device']}")
    run_all(cfg)

if __name__ == "__main__":
    main()
