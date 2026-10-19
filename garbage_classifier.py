import os

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
from torchvision import models

# ----------------------------------------------------
# Settings (safe to import - nothing heavy happens here)
# ----------------------------------------------------
base_path = r"C:\Users\Yash\Desktop\pytorch\Garbage_Classifier\Garbage_Image_Dataset"

train_path = os.path.join(base_path, "train")
val_path = os.path.join(base_path, "validation")
test_path = os.path.join(base_path, "test")

# Model file is saved/loaded next to this script, so training and API agree
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "garbage_resnet18.pth")

# Same order ImageFolder gives (alphabetical)
CLASS_NAMES = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ResNet was trained on ImageNet, so it expects these exact mean/std values
imagenet_mean = [0.485, 0.456, 0.406]
imagenet_std = [0.229, 0.224, 0.225]

# Train: resize + random changes (augmentation) + tensor + normalize
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize(imagenet_mean, imagenet_std),
])

# Validation, test and API: NO random changes, we want clean images
eval_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(imagenet_mean, imagenet_std),
])


# ----------------------------------------------------
# Functions the API will import
# ----------------------------------------------------
def build_model(pretrained=False):
    # pretrained=True  -> download ImageNet weights (used for training)
    # pretrained=False -> empty ResNet18 shape (used when loading our own weights)
    if pretrained:
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    else:
        model = models.resnet18(weights=None)

    # Replace the last layer (1000 ImageNet classes) with len(CLASS_NAMES) outputs
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, len(CLASS_NAMES))

    return model


def load_model(model_path=MODEL_PATH):
    # Build the same shape as training, then fill it with the saved weights
    model = build_model(pretrained=False)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()  # test mode: no learning, just predicting
    return model


def predict_image(model, image):
    # image = a PIL image (RGB). unsqueeze(0) adds the batch dimension
    tensor = eval_transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(tensor)
        probabilities = torch.softmax(outputs, dim=1)[0]  # scores -> percentages

    best_index = torch.argmax(probabilities).item()

    return {
        "prediction": CLASS_NAMES[best_index],
        "confidence": round(probabilities[best_index].item() * 100, 2),
        "all_probabilities": {
            CLASS_NAMES[i]: round(probabilities[i].item() * 100, 2)
            for i in range(len(CLASS_NAMES))
        },
    }


# ----------------------------------------------------
# Training - only runs when you run this file directly
# ----------------------------------------------------
def train():
    # sklearn is only needed for training, so the API doesn't need it installed
    from sklearn.metrics import confusion_matrix
    from sklearn.metrics import classification_report

    print("Using device:", device)

    # ----- STEP 3: Datasets and loaders -----
    train_data = datasets.ImageFolder(root=train_path, transform=train_transform)
    val_data = datasets.ImageFolder(root=val_path, transform=eval_transform)
    test_data = datasets.ImageFolder(root=test_path, transform=eval_transform)

    class_names = train_data.classes
    print("Classes:", class_names)
    print("Train / Val / Test sizes:", len(train_data), len(val_data), len(test_data))

    # Safety check: the API uses CLASS_NAMES, so they must match the folders
    assert class_names == CLASS_NAMES, "Folder classes don't match CLASS_NAMES!"

    batch_size = 16

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    # ----- STEP 4: Model (pretrained ResNet18, last block + last layer are trained) -----
    model = build_model(pretrained=True)

    # Freeze everything first
    for param in model.parameters():
        param.requires_grad = False

    # Un-freeze layer4 (the last ResNet block, learns task-specific features)
    # and fc (the new final layer)
    for param in model.layer4.parameters():
        param.requires_grad = True
    for param in model.fc.parameters():
        param.requires_grad = True

    model = model.to(device)

    # ----- STEP 5: Loss (with softened class weights) and optimizer -----
    # 'trash' has far fewer images, so it gets a bigger weight. The square root keeps
    # the boost gentle (full weights made the model over-predict 'trash').
    counts = torch.bincount(torch.tensor(train_data.targets), minlength=len(CLASS_NAMES)).float()
    class_weights = (counts.sum() / counts).sqrt().to(device)
    class_weights = class_weights / class_weights.mean()
    print("Images per class:", dict(zip(CLASS_NAMES, counts.int().tolist())))

    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # Small learning rate for the pretrained block, bigger one for the new layer
    optimizer = optim.Adam([
        {"params": model.layer4.parameters(), "lr": 1e-4},
        {"params": model.fc.parameters(), "lr": 1e-3},
    ])
    epochs = 5
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # ----- STEP 6: Training loop -----
    best_val_acc = 0.0
    best_state = None

    for epoch in range(epochs):

        # ----- train -----
        model.train()
        train_loss = 0.0
        train_correct = 0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * images.size(0)
            predictions = torch.argmax(outputs, dim=1)
            train_correct += (predictions == labels).sum().item()

        scheduler.step()
        train_loss = train_loss / len(train_data)
        train_acc = train_correct / len(train_data)

        # ----- validation (how the model performs on unseen images) -----
        model.eval()
        val_loss = 0.0
        val_correct = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * images.size(0)
                predictions = torch.argmax(outputs, dim=1)
                val_correct += (predictions == labels).sum().item()

        val_loss = val_loss / len(val_data)
        val_acc = val_correct / len(val_data)

        # Remember the best epoch (by validation accuracy)
        marker = ""
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            marker = "  <- best"

        print(f"Epoch {epoch + 1}/{epochs} | "
              f"Train loss: {train_loss:.4f}, acc: {train_acc:.2%} | "
              f"Val loss: {val_loss:.4f}, acc: {val_acc:.2%}{marker}")

    # Go back to the best epoch before testing and saving
    model.load_state_dict(best_state)
    print(f"\nBest validation accuracy: {best_val_acc:.2%}")

    # ----- STEP 7: Final test on images the model never saw -----
    model.eval()
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)

            outputs = model(images)
            predictions = torch.argmax(outputs, dim=1)

            all_predictions.extend(predictions.cpu().tolist())
            all_labels.extend(labels.tolist())

    correct = sum(p == l for p, l in zip(all_predictions, all_labels))
    print(f"\nTest accuracy: {correct / len(all_labels):.2%}")

    print("\nClassification report:")
    print(classification_report(all_labels, all_predictions, target_names=class_names))

    print("Confusion matrix (rows = real class, columns = predicted class):")
    print(confusion_matrix(all_labels, all_predictions))

    # ----- STEP 8: Save the trained model -----
    torch.save(model.state_dict(), MODEL_PATH)
    print("\nModel saved at:", MODEL_PATH)


# Runs ONLY when you do:  python garbage_classifier.py
# Does NOT run when main.py imports this file
if __name__ == "__main__":
    train()