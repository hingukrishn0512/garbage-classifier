# Garbage Classifier

An image classifier that sorts waste photos into 6 categories: **cardboard, glass, metal, paper, plastic, trash**.
It uses transfer learning with a pretrained **ResNet18** in **PyTorch**, and is served as a **FastAPI** REST API with a simple web page for uploading images.

![Demo]("C:\Users\Yash\Pictures\Screenshots\Screenshot 2026-10-19 134754.png")

## Results

| Metric | Value |
|---|---|
| Test accuracy | **~90%** |
| Classes | 6 |
| Dataset size | 2,527 images |

The model is evaluated on a held-out test set that it never saw during training, using a confusion matrix and per-class precision and recall (printed at the end of training).

## How it works

- **Model:** ImageNet-pretrained ResNet18 with its final layer replaced by a 6-class layer.
- **Fine-tuning:** the last residual block (`layer4`) and the new final layer are trained. All earlier layers stay frozen.
- **Learning rates:** `1e-4` for `layer4` and `1e-3` for the final layer, with a cosine learning-rate schedule over 15 epochs.
- **Augmentation:** random resized crop, horizontal flip, rotation and colour jitter on training images only.
- **Class imbalance:** the `trash` class has far fewer images, so the loss uses softened (square-root) class weights.
- **Best checkpoint:** the epoch with the highest validation accuracy is the one that gets tested and saved.

## Dataset

The data comes from the Kaggle *Garbage Classification* dataset (6 classes, TrashNet-style).

This repository includes the **already split dataset**, so you can train right away without downloading anything from Kaggle. The split was made per class with a fixed random seed (42):

| Split | Images |
|---|---|
| Train (70%) | 1,766 |
| Validation (15%) | 377 |
| Test (15%) | 384 |

Training images per class: cardboard 282, glass 350, metal 287, paper 415, plastic 337, trash 95.

## Project structure

```
Garbage_Classifier/
├── Garbage_Image_Dataset/
│   ├── train/        (cardboard, glass, metal, paper, plastic, trash)
│   ├── validation/   (same 6 folders)
│   └── test/         (same 6 folders)
├── static/
│   └── index.html    web interface
├── garbage_classifier.py   model, training and prediction code
├── main.py                 FastAPI server
├── garbage_resnet18.pth    trained weights
├── requirements.txt
└── README.md
```

## Setup

```bash
git clone <your-repo-url>
cd Garbage_Classifier

python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # macOS / Linux

pip install -r requirements.txt
```

## Run the web app

```bash
uvicorn main:app --reload
```

Then open **http://127.0.0.1:8000**, choose an image and click **Predict**. You get the predicted class and a confidence bar for every class.

## Train the model yourself

```bash
python garbage_classifier.py
```

This trains the model, prints the test accuracy, classification report and confusion matrix, and saves the weights to `garbage_resnet18.pth`. Training on CPU takes a while; a GPU is used automatically if available.

## API

### `POST /predict`

Upload an image as form data with the field name `file`.

```bash
curl -X POST -F "file=@bottle.jpg" http://127.0.0.1:8000/predict
```

Example response:

```json
{
  "prediction": "cardboard",
  "confidence": 88.02,
  "all_probabilities": {
    "cardboard": 88.02,
    "glass": 0.0,
    "metal": 0.04,
    "paper": 9.49,
    "plastic": 0.09,
    "trash": 2.36
  }
}
```

### `GET /classes`

Returns the list of classes the model knows.

### `GET /`

Serves the web interface. Interactive API docs are available at `/docs`.

## Tech stack

Python, PyTorch, torchvision, FastAPI, Uvicorn, scikit-learn, Pillow, HTML/CSS/JavaScript

## Limitations

- The dataset is small and mostly shows single items on plain backgrounds, so accuracy drops on cluttered or real-world photos.
- Glass, metal and plastic can still be confused with each other because they look similar.
- The `trash` class has the fewest images, so its predictions are the least reliable.

## Future improvements

- Unfreeze more layers or try a larger model such as ResNet50 or EfficientNet.
- Add more real-world photos to the training data.
- Add a confidence threshold that returns "not sure" for low-confidence predictions.
