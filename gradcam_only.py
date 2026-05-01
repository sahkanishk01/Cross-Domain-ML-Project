import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from pathlib import Path
import cv2
import matplotlib.pyplot as plt

def build_model(device):
    model = models.resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 2)
    return model.to(device)

def denormalize(img_tensor, mean, std):
    mean = torch.tensor(mean).view(3, 1, 1)
    std = torch.tensor(std).view(3, 1, 1)
    return img_tensor * std + mean

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
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))

    overlay = 0.6 * img + 0.4 * heatmap
    overlay = np.clip(overlay, 0, 1)

    plt.imsave(out_path, overlay)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
best_path = Path("outputs_breast") / "best_resnet_breast.pth"

model = build_model(device)
model.load_state_dict(torch.load(best_path, map_location=device))
model.eval()

val_tf = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

img = Image.open("sample_breast_image.png").convert("RGB")
x = val_tf(img)

save_gradcam_overlay(model, x, Path("outputs_breast") / "gradcam_breast_example.png", device)
print("Saved Grad-CAM")