# Cross-Domain Explainable ML Framework
### Trustworthy AI for Breast Cancer Histopathology and Phishing Website Detection

This project builds a **unified machine learning framework** across two very different domains: medical image classification and cybersecurity tabular classification. Instead of solving only one dataset problem, the project studies how model behavior, explainability, and robustness transfer across domains using a CNN-based breast cancer detector and a Random Forest-based phishing detector.  

## Why this project matters

Most student ML projects stop at accuracy. This project goes further by asking:

- Can one evaluation framework be used across both healthcare and cybersecurity?
- How reliable are the models under perturbations and real-world noise?
- Can predictions be explained in a way that humans can trust?

This makes the work more aligned with real production ML, where **trust, robustness, and explainability** matter as much as raw performance. [web:1504][web:1508][cite:1496]

## Project highlights

- Built a **cross-domain trustworthy ML pipeline** spanning:
  - Breast cancer histopathology image classification
  - Phishing website detection from handcrafted URL/security features
- Used **ResNet50-based CNN** for breast cancer classification on the BreaKHis dataset. [cite:1496]
- Used **Random Forest** for phishing website detection on a 30-feature phishing dataset. [cite:1496]
- Integrated explainability methods:
  - **Grad-CAM / SHAP-style interpretation** for medical images
  - Feature-level importance analysis for phishing prediction. [cite:1496]
- Evaluated **robustness under perturbations** instead of reporting only clean-data accuracy. [cite:1496]

## Architecture

```text
Input Data
│
├── Breast Histopathology Images (BreaKHis)
│   ├── Preprocessing
│   ├── CNN Training (ResNet50)
│   ├── Prediction
│   └── Explainability / Robustness Analysis
│
└── Phishing Website Features
    ├── Feature Preprocessing
    ├── Random Forest Training
    ├── Prediction
    └── Feature Importance / Robustness Analysis
```

## Datasets

### 1) Breast Cancer Histopathology
- Dataset: **BreaKHis**
- Task: Binary breast cancer classification
- Modality: Histopathology microscopy images. [cite:1496][cite:1497]

### 2) Phishing Website Detection
- Task: Binary classification of phishing vs legitimate websites
- Input: 30 engineered website/URL/security features such as `having_IP_Address`, `URL_Length`, `SSLfinal_State`, `Request_URL`, and `Google_Index`. [cite:1496]

## Models used

| Domain | Model | Input Type | Goal |
|---|---|---|---|
| Breast Cancer | ResNet50-based CNN | Images | Detect benign vs malignant patterns |
| Phishing Detection | Random Forest | Tabular features | Detect phishing vs legitimate websites |

This combination shows the ability to work across both **deep learning for vision** and **classical ML for structured security data**, which is valuable for applied ML roles. [cite:1496][web:1504]

## Explainability

A major focus of this project is not just prediction, but **interpretable prediction**.

### Breast Cancer
The breast cancer model was analyzed using explainability techniques to identify image regions influencing predictions. Important visual patterns included nuclei density and glandular disruption, helping connect model behavior to medically meaningful structures. [cite:1496]

### Phishing Detection
The phishing model was interpreted through feature-level importance. Features such as IP usage, URL length, and SSL-related signals were identified as influential for detecting suspicious websites. [cite:1496]

## Robustness analysis

To simulate real-world deployment challenges, both models were tested under perturbations and compared against their clean-data performance.

- Breast CNN robustness drop: **7.2%**
- Phishing RF robustness drop: **5.8%** [cite:1496]

This analysis is important because a model that performs well only on ideal data is not enough for healthcare or cybersecurity use cases. [cite:1496][web:1504]

## Key contributions

- Designed a single comparative framework across **two unrelated domains**
- Combined **accuracy + explainability + robustness** into one pipeline
- Demonstrated practical understanding of:
  - Computer vision
  - Tabular ML
  - Trustworthy AI
  - Evaluation under perturbations
  - Reproducible experimentation. [cite:1496][web:1506]

## Tech stack

- Python
- PyTorch
- TensorFlow / Keras
- Scikit-learn
- NumPy
- Pandas
- SHAP
- Grad-CAM
- Joblib
- Matplotlib / Seaborn. [cite:1496]

## Repository structure

```text
Cross-Domain-ML-Project/
│
├── breast_pipeline.py
├── phishing_pipeline.py
├── demo_prediction.py
├── requirements.txt
├── README.md
│
├── outputs_breast/
├── outputs_phishing/
├── sample_breast_image.png
└── ...
```

## How to run

### 1. Clone the repository

```bash
git clone https://github.com/sahkanishk01/Cross-Domain-ML-Project.git
cd Cross-Domain-ML-Project
```

### 2. Create virtual environment

```bash
python -m venv venv
venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run demo prediction

```bash
python demo_prediction.py
```

## Demo output

The demo script performs:
- Breast cancer image prediction using a saved CNN model
- Phishing website prediction using a saved Random Forest model

Example outputs may include:
- `Breast Prediction: benign`
- Confidence score for breast image prediction
- Phishing prediction using the 30-feature input sample

## Business and research impact

This project demonstrates the ability to build ML systems that are not limited to one benchmark or one architecture. It shows readiness for roles involving applied machine learning, computer vision, cybersecurity analytics, healthcare AI, and trustworthy AI research. [web:1504][web:1508][cite:1496]

For recruiters and reviewers, the strongest signal here is not just model training — it is the ability to:
- frame a problem well,
- choose domain-appropriate models,
- evaluate trustworthiness,
- explain decisions clearly,
- and build a reusable research pipeline. [web:1504][web:1506]

## Future improvements

- Add Streamlit or Flask web interface for live demo
- Add model cards and data cards
- Add adversarial robustness experiments
- Add CI/CD checks and automated evaluation
- Export explainability reports for both domains. [web:1508][web:1511]

## Author

**Kanishk Sah**  
Final Year Project – Cross-Domain Generalization of ML Models

## License

This project is for academic and portfolio use. Add an MIT License if you want others to reuse the code.
