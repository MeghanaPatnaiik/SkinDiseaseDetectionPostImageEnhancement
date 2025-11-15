import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import kagglehub
from PIL import Image

import download_model
download_model.download_models()

# ---------------------------
# 1. Download dataset
# ---------------------------
print(" Downloading dataset from Kaggle...")
path = kagglehub.dataset_download("shubhamgoel27/dermnet")
base_dir = os.path.join(path, "DermNet")

# For DermNet, structure is:
# DermNet/train
# DermNet/test
test_dir = os.path.join(base_dir, "test")

if not os.path.exists(test_dir):
    raise FileNotFoundError(" Test folder not found. Dataset structure may have changed.")
else:
    print(f" Test folder found: {test_dir}")

# ---------------------------
# 2. Device setup
# ---------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\nUsing device: {device}")

# ---------------------------
# 3. Preprocessing
# ---------------------------
data_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

test_dataset = datasets.ImageFolder(root=test_dir, transform=data_transforms)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
class_names = test_dataset.classes

# ---------------------------
# 4. Function to Evaluate Model
# ---------------------------
def evaluate_model(model, loader):
    model.eval()
    correct, total = 0, 0

    with torch.no_grad():
        for inputs, labels in tqdm(loader, desc="Testing"):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return 100 * correct / total

# ---------------------------
# 5. Load Models
# ---------------------------
print("\n Loading ResNet50 model...")
model_resnet50 = torch.load("resnet50_skin_model.pth", map_location=device)
model_resnet50.to(device)
print(" ResNet50 loaded!")

print("\n Loading Best ResNet50 model...")
model_best = torch.load("best_resnet50_skin.pth", map_location=device)
model_best.to(device)
print(" Best ResNet50 loaded!")

# ---------------------------
# 6. Evaluate Models
# ---------------------------
print("\n Evaluating ResNet50 on test set...")
acc_resnet50 = evaluate_model(model_resnet50, test_loader)
print(f" ResNet50 Test Accuracy: {acc_resnet50:.2f}%\n")

print("\n Evaluating Best ResNet50 on test set...")
acc_best = evaluate_model(model_best, test_loader)
print(f" Best ResNet50 Test Accuracy: {acc_best:.2f}%\n")

# ---------------------------
# 7. Example Prediction
# ---------------------------
example_img = os.path.join(test_dir, class_names[0],
                           os.listdir(os.path.join(test_dir, class_names[0]))[0])
print(f" Example image used: {example_img}")

img = Image.open(example_img).convert("RGB")
img_t = data_transforms(img).unsqueeze(0).to(device)

# --- ResNet50 prediction ---
out50 = model_resnet50(img_t)
_, pred50 = torch.max(out50, 1)
print(f"\n ResNet50 Prediction: {class_names[pred50.item()]}")

# --- Best Model prediction ---
outB = model_best(img_t)
_, predB = torch.max(outB, 1)
print(f" Best ResNet50 Prediction: {class_names[predB.item()]}")
