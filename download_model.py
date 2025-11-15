import gdown
import os

# ---------------------------------------------------
#  Model Weights to Download 
# ---------------------------------------------------
MODELS = {
    "resnet50_enhanced_skin_disease_final.pth": "1SjcfSqf_TzJRtEcA4nzs-MJni43AiaTq"
}

def download_models():
    """
    Downloads model weight files from Google Drive using gdown.
    Skips files already present in the directory.
    """
    for model_name, file_id in MODELS.items():
        if not os.path.exists(model_name):
            print(f"\n Downloading {model_name}...")
            url = f"https://drive.google.com/uc?id={file_id}"
            gdown.download(url, model_name, quiet=False)
            print(f" {model_name} downloaded successfully.\n")
        else:
            print(f" {model_name} already exists. Skipping.\n")

if __name__ == "__main__":
    download_models()
