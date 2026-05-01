# breastpipeline.py

import os
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, recall_score
from sklearn.model_selection import train_test_split
from PIL import Image
from tqdm import tqdm
import cv2
import matplotlib.pyplot as plt

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")  

def set_seed(seed=42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def compute_metrics(y_true, y_pred, y_score):
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    try:
        auc = roc_auc_score(y_true, y_score)
    except ValueError:
        auc = float("nan")
    rec = recall_score(y_true, y_pred)
    return acc, f1, auc, rec

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

class SubsetWithTransform(torch.utils.data.Dataset):
    def __init__(self, base_ds, idxs, transform):
        self.base_ds = base_ds
        self.idxs = idxs
        self.transform = transform

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, i):
        x, y = self.base_ds[self.idxs[i]]
        x = self.transform(x)
        return x, y

def build_model(device):
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    model.fc = nn.Linear(model.fc.in_features, 2)
    return model.to(device)

def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    for imgs, labels in tqdm(loader, desc="Train"):
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        out = model(imgs)
        loss = criterion(out, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * imgs.size(0)
    return total_loss / len(loader.dataset)

@torch.no_grad()
def eval_epoch(model, loader, device):
    model.eval()
    ys, yp, yscores = [], [], []
    for imgs, labels in tqdm(loader, desc="Val"):
        imgs, labels = imgs.to(device), labels.to(device)
        logits = model(imgs)
        probs = torch.softmax(logits, dim=1)[:, 1]
        preds = (probs >= 0.5).long()
        ys.extend(labels.cpu().numpy())
        yp.extend(preds.cpu().numpy())
        yscores.extend(probs.cpu().numpy())
    return compute_metrics(np.array(ys), np.array(yp), np.array(yscores))

def add_gaussian_noise(imgs, sigma=0.1):
    noise = torch.randn_like(imgs) * sigma
    return torch.clamp(imgs + noise, 0, 1)

def occlude(imgs, frac=0.1):
    b, c, h, w = imgs.shape
    oh, ow = int(h * np.sqrt(frac)), int(w * np.sqrt(frac))
    out = imgs.clone()
    for i in range(b):
        y = np.random.randint(0, max(1, h - oh))
        x = np.random.randint(0, max(1, w - ow))
        out[i, :, y:y+oh, x:x+ow] = 0
    return out

class GradCAM:
    def __init__(self, model, target_layer_name="layer4"):
        self.model = model
        self.model.eval()
        self.gradients = None
        self.activations = None
        target_layer = dict(self.model.named_modules())[target_layer_name]
        target_layer.register_forward_hook(self.forward_hook)
        target_layer.register_full_backward_hook(self.backward_hook)

    def forward_hook(self, module, inp, out):
        self.activations = out

    def backward_hook(self, module, grad_in, grad_out):
        self.gradients = grad_out[0]

    def __call__(self, x, class_idx=None):
        self.model.zero_grad()
        out = self.model(x)
        if class_idx is None:
            class_idx = out.argmax(dim=1).item()
        score = out[:, class_idx].sum()
        score.backward()
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1)
        cam = torch.relu(cam)[0].detach().cpu().numpy()
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        return cam

def denormalize(img_tensor, mean, std):
            mean = torch.tensor(mean).view(3, 1, 1)
            std = torch.tensor(std).view(3, 1, 1)
            return img_tensor * std + mean

def save_gradcam_overlay(model, img_tensor, out_path, device):
    cam_gen = GradCAM(model, "layer4")
    x = img_tensor.unsqueeze(0).to(device)
    cam = cam_gen(x)

    img = denormalize(
        img_tensor.cpu(),
        (0.4914, 0.4822, 0.4465),
        (0.2023, 0.1994, 0.2010)
    )
    img = img.permute(1, 2, 0).numpy().astype(np.float32)
    img = np.clip(img, 0, 1)

    heatmap = cv2.applyColorMap((cam * 255).astype(np.uint8), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

    overlay = 0.6 * img + 0.4 * heatmap
    overlay = np.clip(overlay, 0, 1)

    plt.imsave(out_path, overlay)

@torch.no_grad()
def eval_with_transform(model, loader, transform_fn, device):
    model.eval()
    ys, yp, yscores = [], [], []
    for imgs, labels in tqdm(loader, desc="Robust"):
        imgs, labels = imgs.to(device), labels.to(device)
        imgs = transform_fn(imgs)
        logits = model(imgs)
        probs = torch.softmax(logits, dim=1)[:, 1]
        preds = (probs >= 0.5).long()
        ys.extend(labels.cpu().numpy())
        yp.extend(preds.cpu().numpy())
        yscores.extend(probs.cpu().numpy())
    return compute_metrics(np.array(ys), np.array(yp), np.array(yscores))


def predict_single_breast_image(img_path, model_path, device, val_tf):
    model = build_model(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    img = Image.open(img_path).convert("RGB")
    x = val_tf(img).unsqueeze(0).to(device)

    with torch.no_grad():
        out = model(x)
        probs = torch.softmax(out, dim=1)[0]
        pred = int(torch.argmax(probs).item())

    classes = ["benign", "malignant"]
    print("Breast Prediction:", classes[pred], "Confidence:", float(probs[pred]))
    return classes[pred], float(probs[pred])

def main():
    set_seed(42)
    device = get_device()

    BREAST_ROOT = Path(r"D:\Cross domain of ml models\BreakHis_binary")
    OUTPUTDIR = Path("outputs_breast")
    ensure_dir(OUTPUTDIR)

    train_tf = transforms.Compose([
        transforms.RandomRotation(90),
        transforms.RandomHorizontalFlip(),
        transforms.RandomResizedCrop(224),
        transforms.ColorJitter(0.4, 0.4, 0.4, 0.1),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    val_tf = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    if BREAST_ROOT.exists():
        full_ds = datasets.ImageFolder(root=str(BREAST_ROOT), transform=None)   
        idxs = np.arange(len(full_ds))
        train_idx, val_idx = train_test_split(
            idxs, test_size=0.2, stratify=full_ds.targets, random_state=42
        )
        train_ds = SubsetWithTransform(full_ds, train_idx, train_tf)
        val_ds = SubsetWithTransform(full_ds, val_idx, val_tf)

        train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=4)
        val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=4)

        model = build_model(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

        best_acc = 0.0
        best_path = OUTPUTDIR / "best_resnet_breast.pth"

        for ep in range(3):
            loss = train_epoch(model, train_loader, criterion, optimizer, device)
            acc, f1, auc, rec = eval_epoch(model, val_loader, device)
            print(f"Epoch {ep+1}/3 - loss={loss:.4f} acc={acc:.4f} f1={f1:.4f} auc={auc:.4f} rec={rec:.4f}")
            if acc > best_acc:
                best_acc = acc
                torch.save(model.state_dict(), best_path)

        model.load_state_dict(torch.load(best_path, map_location=device))
        model.eval()

        accb, f1b, aucb, recb = eval_epoch(model, val_loader, device)
        print("FINAL BREAST METRICS:", accb, f1b, aucb, recb)

        imgs, labels = next(iter(val_loader))

        img = denormalize(
            imgs[0].cpu(),
            (0.4914, 0.4822, 0.4465),
            (0.2023, 0.1994, 0.2010)
        )
        img = img.permute(1, 2, 0).numpy()
        img = np.clip(img, 0, 1)
        plt.imsave(OUTPUTDIR / "breast_sample.png", img)
        print("Saved breast_sample.png")

        save_gradcam_overlay(model, imgs[0], OUTPUTDIR / "gradcam_breast_example.png", device)
        print("Saved Grad-CAM example to", OUTPUTDIR / "gradcam_breast_example.png")


    else:
        print("BreaKHis folder not found, skipping training demo.")

    sample_img = "sample_breast_image.png"
    if Path(sample_img).exists() and (OUTPUTDIR / "best_resnet_breast.pth").exists():
        predict_single_breast_image(sample_img, OUTPUTDIR / "best_resnet_breast.pth", device, val_tf)

if __name__ == "__main__":
    main()