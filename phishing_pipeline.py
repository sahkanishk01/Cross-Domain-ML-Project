import os
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, recall_score
import joblib
import shap
import matplotlib.pyplot as plt
from ucimlrepo import fetch_ucirepo

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def set_seed(seed=42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def compute_metrics(y_true, y_pred, y_score):
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    y_score = np.asarray(y_score, dtype=float)

    print("compute_metrics unique y_true:", np.unique(y_true))
    print("compute_metrics unique y_pred:", np.unique(y_pred))

    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, pos_label=1, average="binary", zero_division=0)
    try:
        auc = roc_auc_score(y_true, y_score)
    except ValueError:
        auc = float("nan")
    rec = recall_score(y_true, y_pred, pos_label=1, average="binary", zero_division=0)
    return acc, f1, auc, rec

def perturb_url_features(df):
    dfp = df.copy()
    if "havingIPAddress" in dfp.columns:
        dfp["havingIPAddress"] = 0
    if "URLLength" in dfp.columns:
        dfp["URLLength"] = 0
    if "SSLfinalState" in dfp.columns:
        dfp["SSLfinalState"] = 1
    if "prefixSuffix" in dfp.columns:
        dfp["prefixSuffix"] = 0
    return dfp

class PhishingCNN(nn.Module):
    def __init__(self, in_features):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveMaxPool1d(1)
        self.fc = nn.Linear(64, 2)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = self.pool(x).squeeze(-1)
        x = self.fc(x)
        return x

def train_ph_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        out = model(xb)
        loss = criterion(out, yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * xb.size(0)
    return total_loss / len(loader.dataset)

@torch.no_grad()
def eval_ph_epoch(model, loader, device):
    model.eval()
    ys, yp, yscores = [], [], []
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        logits = model(xb)
        probs = torch.softmax(logits, dim=1)[:, 1]
        preds = (probs >= 0.5).long()
        ys.extend(yb.cpu().numpy())
        yp.extend(preds.cpu().numpy())
        yscores.extend(probs.cpu().numpy())
    return compute_metrics(np.array(ys), np.array(yp), np.array(yscores))

import joblib
import pandas as pd

def predict_single_phishing_sample(sample_dict, model_path):
    model = joblib.load(model_path)

    sample_df = pd.DataFrame([sample_dict])

    prediction = model.predict(sample_df)[0]

    if hasattr(model, "predict_proba"):
        probability = model.predict_proba(sample_df)[0][1]
    else:
        probability = None

    label = "Phishing" if prediction == 1 else "Legitimate"

    print("\n--- Phishing Demo Prediction ---")
    print("Input sample:", sample_dict)
    print("Predicted class:", prediction)
    print("Predicted label:", label)
    if probability is not None:
        print(f"Phishing probability: {probability:.4f}")

    return prediction, probability, label

def run_shap_and_robustness(rf_model, X_train, X_test, y_test, feature_names, outputdir):
    ensure_dir(outputdir)
    X_test_df = pd.DataFrame(X_test, columns=feature_names)

    explainer = shap.Explainer(rf_model, X_test_df)
    idx = np.random.choice(len(X_test_df), size=min(500, len(X_test_df)), replace=False)
    X_shap = X_test_df.iloc[idx]

    shap_exp = explainer(X_shap)

    print("SHAP values shape:", shap_exp.values.shape)
    print("X_shap shape:", X_shap.shape)

    plt.figure(figsize=(10, 6))

    if len(shap_exp.values.shape) == 3:
        shap.summary_plot(
            shap_exp.values[:, :, 1],
            X_shap,
            show=False
    )
    else:
        shap.summary_plot(
        shap_exp.values,
        X_shap,
        show=False
    )

    plt.tight_layout()
    plt.savefig(Path(outputdir) / "shap_phishing_summary.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("Saved SHAP summary to", Path(outputdir) / "shap_phishing_summary.png")

    prob_clean = rf_model.predict_proba(X_test_df)[:, 1]
    pred_clean = (prob_clean >= 0.5).astype(int)
    acc_clean, f1_clean, auc_clean, rec_clean = compute_metrics(y_test, pred_clean, prob_clean)

    X_pert = perturb_url_features(X_test_df)
    prob_pert = rf_model.predict_proba(X_pert)[:, 1]
    pred_pert = (prob_pert >= 0.5).astype(int)
    acc_pert, f1_pert, auc_pert, rec_pert = compute_metrics(y_test, pred_pert, prob_pert)

    print("PHISHING CLEAN:", acc_clean, f1_clean, auc_clean, rec_clean)
    print("PHISHING PERT :", acc_pert, f1_pert, auc_pert, rec_pert)
    print("AUC DROP:", auc_clean - auc_pert)
    print("REC DROP:", rec_clean - rec_pert)

def main():
    set_seed(42)
    device = get_device()
    OUTPUTDIR = Path("outputs_phishing")
    ensure_dir(OUTPUTDIR)

    phishing = fetch_ucirepo(id=327)

    X = phishing.data.features
    y = phishing.data.targets.values.ravel().astype(int)

    # convert labels from {-1, 1} to {0, 1}
    y = np.where(y == -1, 0, 1)

    print("Unique y:", np.unique(y))

    X_train, X_test, y_train, y_test = train_test_split(
        X.values,
        y,
        test_size=0.2,
        stratify=y,
        random_state=42
    )

    print("Unique y_train:", np.unique(y_train))
    print("Unique y_test:", np.unique(y_test))

    lr = LogisticRegression(max_iter=500)
    lr.fit(X_train, y_train)

    prob_lr = lr.predict_proba(X_test)[:, 1]
    pred_lr = (prob_lr >= 0.5).astype(int)

    print("Unique pred_lr:", np.unique(pred_lr))

    acc_lr, f1_lr, auc_lr, rec_lr = compute_metrics(y_test, pred_lr, prob_lr)
    print("LR metrics:", acc_lr, f1_lr, auc_lr, rec_lr)

    rf = RandomForestClassifier(n_estimators=300, max_depth=None, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    prob_rf = rf.predict_proba(X_test)[:, 1]
    pred_rf = (prob_rf >= 0.5).astype(int)
    acc_rf, f1_rf, auc_rf, rec_rf = compute_metrics(y_test, pred_rf, prob_rf)
    print("RF metrics:", acc_rf, f1_rf, auc_rf, rec_rf)

    joblib.dump(rf, OUTPUTDIR / "rf_phishing.pkl")

    feature_names = phishing.data.features.columns
    run_shap_and_robustness(rf, X_train, X_test, y_test, feature_names, OUTPUTDIR)

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    X_train_t = torch.tensor(X_train_s, dtype=torch.float32).unsqueeze(1)
    X_test_t = torch.tensor(X_test_s, dtype=torch.float32).unsqueeze(1)
    y_train_t = torch.tensor(y_train, dtype=torch.long)
    y_test_t = torch.tensor(y_test, dtype=torch.long)

    train_ds = torch.utils.data.TensorDataset(X_train_t, y_train_t)
    test_ds = torch.utils.data.TensorDataset(X_test_t, y_test_t)

    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False)

    phcnn = PhishingCNN(in_features=X_train.shape[1]).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(phcnn.parameters(), lr=1e-3)

    best_acc = 0.0
    for ep in range(10):
        loss = train_ph_epoch(phcnn, train_loader, criterion, optimizer, device)
        acc, f1, auc, rec = eval_ph_epoch(phcnn, test_loader, device)
        print(f"CNN Epoch {ep+1}/10 - loss={loss:.4f} acc={acc:.4f} f1={f1:.4f} auc={auc:.4f} rec={rec:.4f}")
        if acc > best_acc:
            best_acc = acc
            torch.save(phcnn.state_dict(), OUTPUTDIR / "best_phishing_cnn.pth")

    phcnn.load_state_dict(torch.load(OUTPUTDIR / "best_phishing_cnn.pth", map_location=device))
    acc_cnn, f1_cnn, auc_cnn, rec_cnn = eval_ph_epoch(phcnn, test_loader, device)
    print("FINAL PHISHING CNN METRICS:", acc_cnn, f1_cnn, auc_cnn, rec_cnn)

if __name__ == "__main__":
    main()