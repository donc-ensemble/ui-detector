# UI Widget Detection for Smart TV QA Automation

## Overview
This project focuses on using ML-based CNN frameworks for UI widget detection in media TV apps, replacing LLM-based approaches for efficiency and cost-effectiveness. It leverages YOLOv8 for image-based detection and PaddleOCR for text recognition.

## Features
- Detects common UI widgets: **buttons, text inputs, cards, qr code**.
- Identifies loading indicators to determine when a page is still loading.
- Identifies the state of each element with focused indicator.
- Uses a **video recording of Streams frontend** as input for training and predictions.

## Tech Stack
- **Python** 3.11.7
- **YOLOv8** (for visual detection)
- **PaddleOCR** (for text recognition in UI elements)
- **CVAT** (for dataset labeling)

## Setup Instructions
### 1. Create Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

## Dataset Preparation
1. **Use CVAT** (https://app.cvat.ai) to manually label UI components in the dataset.
2. Export labeled data in YOLO format.
3. The folder names `data/images/` and `data/labels/` must be strictly implemented.
4. Place the raw images `data/images/train`.
5. Place the label texts from the exported YOLOV8 file from CVAT into `data/labels/train`
6. Place the dataset in `data/` and update `config.yaml` accordingly.

## Training the Model
To train a new YOLOv8 model on the labeled dataset:
```bash
python train.py
```

## Running Predictions
To perform UI widget detection on a video or image input, place the UI video into videos folder and name it to `stream.mp4` otherwise modify the main entrypoint:
```bash
python predict.py
```
## Output:
* `json/`: json log of the detected elements per frame on every state change.
* `videos/`: annotated ppocr video.

## Repository Link
For more details, refer to the project repository:
[AI TV Widget Detection](https://stash.ensemble.com/projects/ENSAI/repos/ai-tv-widget-detection/browse)

