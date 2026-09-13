#!/usr/bin/env python3
import argparse
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from PIL import Image
from src.models import build_model
from src.data import GTSRB_MEAN, GTSRB_STD, get_class_name
from src.inference import predict_image


def main():
    p = argparse.ArgumentParser(description="Predict a GTSRB traffic-sign image")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--image", required=True)
    p.add_argument("--model", default="resnet18", choices=["resnet18", "baseline_cnn"])
    p.add_argument("--image-size", type=int, default=64)
    p.add_argument("--top-k", type=int, default=5)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(args.model, 43, pretrained=False).to(device)
    state = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state["model_state"])
    model.eval()

    image = Image.open(args.image).convert("RGB")
    results = predict_image(model, device, image, image_size=args.image_size, top_k=args.top_k)

    print(f"Top-{args.top_k} predictions:")
    for class_idx, class_name, confidence in results:
        print(f"  {class_name} (class {class_idx}): {confidence:.2%}")


if __name__ == "__main__":
    main()
