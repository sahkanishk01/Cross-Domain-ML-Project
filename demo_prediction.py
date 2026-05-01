# demo_prediction.py

from pathlib import Path
from breast_pipeline import get_device, predict_single_breast_image, set_seed
from phishing_pipeline import predict_single_phishing_sample
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

    phishing_model = Path("outputs_phishing/rf_phishing.pkl")
    if phishing_model.exists():
        sample = {
        "having_IP_Address": -1,
        "URL_Length": 1,
        "Shortining_Service": 1,
        "having_At_Symbol": 1,
        "double_slash_redirecting": -1,
        "Prefix_Suffix": -1,
        "having_Sub_Domain": -1,
        "SSLfinal_State": 1,
        "Domain_registeration_length": -1,
        "Favicon": 1,
        "port": 1,
        "HTTPS_token": -1,
        "Request_URL": 1,
        "URL_of_Anchor": -1,
        "Links_in_tags": 1,
        "SFH": -1,
        "Submitting_to_email": -1,
        "Abnormal_URL": -1,
        "Redirect": 0,
        "on_mouseover": 1,
        "RightClick": 1,
        "popUpWidnow": 1,
        "Iframe": 1,
        "age_of_domain": -1,
        "DNSRecord": -1,
        "web_traffic": -1,
        "Page_Rank": -1,
        "Google_Index": 1,
        "Links_pointing_to_page": 1,
        "Statistical_report": -1
        }
        predict_single_phishing_sample(sample, str(phishing_model))
       
    else:
         print("Phishing model not found.")

if __name__ == "__main__":
    main()