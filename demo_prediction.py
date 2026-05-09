# demo_prediction.py

from pathlib import Path
from breast_pipeline import get_device, predict_single_breast_image, set_seed
import pandas as pd
import joblib

def main():
    set_seed(42)
    device = get_device()

    breast_model = Path("outputs_breast/best_resnet_breast.pth")
    breast_img = Path("sample_breast_image.png")

    if breast_model.exists() and breast_img.exists():
        from breast_pipeline import transforms
        val_tf = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        predict_single_breast_image(str(breast_img), breast_model, device, val_tf)
    else:
        print("Breast demo files not found.")

if __name__ == "__main__":
    main()
