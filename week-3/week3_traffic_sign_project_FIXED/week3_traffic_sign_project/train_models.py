#!/usr/bin/env python3
"""
Training script to run all experiments and generate models.
This creates trained models and visualizations for the Streamlit app.

Usage:
    python train_models.py
    python train_models.py --config configs/experiments.yaml
"""

import argparse
from pathlib import Path
import sys
import logging

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.config import load_config
from src.experiments import run_all
from src.utils import seed_everything, choose_device

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Run controlled GTSRB traffic-sign experiments"
    )
    parser.add_argument(
        "--config",
        default="configs/experiments.yaml",
        help="Path to experiments config file"
    )
    parser.add_argument(
        "--output",
        default="outputs",
        help="Output directory for results and models"
    )
    
    args = parser.parse_args()
    
    # Load configuration
    config_path = Path(args.config)
    if not config_path.exists():
        logger.error(f"Config file not found: {config_path}")
        sys.exit(1)
    
    logger.info(f"Loading config from: {config_path}")
    cfg = load_config(str(config_path))
    
    # Update output directory if specified
    if args.output != "outputs":
        cfg["output_dir"] = Path(args.output)
    
    # Setup environment
    seed_everything(cfg.get("seed", 42))
    cfg["device"] = choose_device(cfg.get("device", "auto"))
    
    logger.info(f"Device: {cfg['device']}")
    logger.info(f"Output directory: {cfg['output_dir']}")
    logger.info(f"Running {len(cfg['experiments'])} experiments...")
    
    # Create output directory
    Path(cfg["output_dir"]).mkdir(parents=True, exist_ok=True)
    
    try:
        # Run all experiments
        results_df = run_all(cfg)
        
        logger.info("=" * 60)
        logger.info("Training completed successfully!")
        logger.info("=" * 60)
        logger.info("\nResults Summary:")
        logger.info(results_df[["experiment", "accuracy", "precision_macro", "recall_macro", "f1_macro"]].to_string(index=False))
        logger.info("\nGenerated files:")
        output_dir = Path(cfg["output_dir"])
        for exp_dir in sorted(output_dir.iterdir()):
            if exp_dir.is_dir() and exp_dir.name != ".gitkeep":
                logger.info(f"\n  Experiment: {exp_dir.name}")
                logger.info(f"    - Model: {exp_dir / 'best_model.pt'}")
                logger.info(f"    - Metrics: {exp_dir / 'metrics.json'}")
                logger.info(f"    - Training curves: {exp_dir / 'training_curves.png'}")
                logger.info(f"    - Confusion matrix: {exp_dir / 'confusion_matrix.png'}")
                logger.info(f"    - ROC curves: {exp_dir / 'roc_auc.png'}")
        
        logger.info(f"\n  Comparison results: {output_dir / 'results.csv'}")
        logger.info(f"  Comparison chart: {output_dir / 'experiment_comparison.png'}")
        
        logger.info("\n" + "=" * 60)
        logger.info("Ready to use Streamlit app!")
        logger.info("Run: streamlit run app.py")
        logger.info("=" * 60)
        
    except Exception as e:
        logger.error(f"Training failed with error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
