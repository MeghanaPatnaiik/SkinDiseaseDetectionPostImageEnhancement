import os
import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from tqdm import tqdm
import kagglehub
from PIL import Image

# your model download file
import download_model
download_model.download_models()

# ---------------------------
# 1. Download dataset
# ---------------------------
print("\nDownloading dataset from Kaggle...")
path = kagglehub.dataset_download("shubhamgoel27/dermnet")

base_dir = path
test_dir = os.path.join(base_dir, "test")

if not os.path.exists(test_dir):
    print("\n test/ folder not found. Contents of dataset root:")
    print(os.listdir(base_dir))
    raise FileNotFoundError(" Dataset structure changed. Could not find test folder.")

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
print(f"Found {len(class_names)} classes.")

# ---------------------------
# 4. Evaluate Function
# ---------------------------
def evaluate_model(model, loader):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in tqdm(loader, desc="Testing"):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return 100 * correct / total


# ---------------------------
# 5. Load ONLY ONE MODEL
# ---------------------------
num_classes = len(class_names)

print("\nLoading Enhanced ResNet50 model...")

model_resnet50 = models.resnet50(weights=None)

# FIX ARCHITECTURE TO MATCH YOUR SAVED STATE
num_ftrs = model_resnet50.fc.in_features
model_resnet50.fc = nn.Sequential(
    nn.Dropout(0.5),
    nn.Linear(num_ftrs, num_classes)
)

state_dict = torch.load("resnet50_enhanced_skin_disease_final.pth",
                        map_location=device)

model_resnet50.load_state_dict(state_dict)
model_resnet50.to(device)
print(" Enhanced ResNet50 loaded.\n")

# ---------------------------
# 6. Evaluate
# ---------------------------
print("\nEvaluating Enhanced ResNet50...")
acc = evaluate_model(model_resnet50, test_loader)
print(f"Enhanced ResNet50 Accuracy: {acc:.2f}%")

# ---------------------------
# 7. Example Prediction
# ---------------------------
example_img = os.path.join(test_dir, class_names[0],
                           os.listdir(os.path.join(test_dir, class_names[0]))[0])

print(f"\nExample Image: {example_img}")

img = Image.open(example_img).convert("RGB")
img_t = data_transforms(img).unsqueeze(0).to(device)

# Prediction
out = model_resnet50(img_t)
_, pred = torch.max(out, 1)

print(f"\nPrediction: {class_names[pred.item()]}")
